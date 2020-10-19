// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <fstream>
#include <aloam_velodyne/common.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>

#include "lidarFactor.hpp"
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "aloam_velodyne/ClusterArray.h"

#include "kkl/alg/dp_means.hpp"
#include "hdl_people_detection/marcel_people_detector.hpp"
#include "hdl_people_tracking/people_tracker.hpp"

static int frameCount = 0;

static double timeLaserCloudCornerLast = 0;
static double timeLaserCloudSurfLast = 0;
static double timeLaserCloudFullRes = 0;
static double timeLaserOdometry = 0;


static int laserCloudCenWidth = 10;
static int laserCloudCenHeight = 10;
static int laserCloudCenDepth = 5;
const int laserCloudWidth = 21;
const int laserCloudHeight = 21;
const int laserCloudDepth = 11;


const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth; //4851


static int laserCloudValidInd[125];
static int laserCloudSurroundInd[125];

// input: from odom
static pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
static pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());

// ouput: all visualble cube points
static pcl::PointCloud<PointType>::Ptr laserCloudSurround(new pcl::PointCloud<PointType>());

// surround points in map to build tree
static pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap(new pcl::PointCloud<PointType>());
static pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap(new pcl::PointCloud<PointType>());

//input & output: points in one frame. local --> global
static pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

// points in every cube
static pcl::PointCloud<PointType>::Ptr laserCloudCornerArray[laserCloudNum];
static pcl::PointCloud<PointType>::Ptr laserCloudSurfArray[laserCloudNum];

// 带动态标签的当前帧点云
static pcl::PointCloud<PointType>::Ptr currDynamicCloudStack(new pcl::PointCloud<PointType>());

//kd-tree
static pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());
static pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());

static double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
static Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);
static Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);

// wmap_T_odom * odom_T_curr = wmap_T_curr;
// transformation between odom's world and map's world frame
static Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
static Eigen::Vector3d t_wmap_wodom(0, 0, 0);

static Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
static Eigen::Vector3d t_wodom_curr(0, 0, 0);


static std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLastBuf;
static std::queue<sensor_msgs::PointCloud2ConstPtr> surfLastBuf;
static std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
static std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
static std::mutex mBuf;

static pcl::VoxelGrid<PointType> downSizeFilterCorner;
static pcl::VoxelGrid<PointType> downSizeFilterSurf;

static std::vector<int> pointSearchInd;
static std::vector<float> pointSearchSqDis;

static PointType pointOri, pointSel;

static ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes, pubOdomAftMapped, pubOdomAftMappedHighFrec, pubLaserAfterMappedPath;
static ros::Publisher pubStaticCloudMap,pubDynamicCloudCurr;

static nav_msgs::Path laserAfterMappedPath;

static Accumulator<float> connerOpticalDistanceMean; //光流距离累加器
static Accumulator<float> surfOpticalDistanceMean; //光流距离累加器
static Accumulator<float> objectSpeedMean; //聚类速度累加器
static Accumulator<float> tarckerSpeedMean; //跟踪速度累加器
static Accumulator<float> removeTimeMean; //运行时间累加器

static std::unique_ptr<hdl_people_tracking::PeopleTracker> tracker;

static pcl::PointCloud<PointType>::Ptr staticCloudMap(new pcl::PointCloud<PointType>());

static float lineRes = 0.1f;
static float planeRes = 0.1f;
static float maxObjectSpeed = 0;
static float maxClassDist = 0;
static int min_pts = 10;
static int max_pts = 8192;
static float cluster_min_size = 0.2f;
static float cluster_max_size = 2.0;
static float cluster_tolerane = 0.5;
static float cluster_lambda = 0.5;

static bool autoMapping = true;
static float autoMappingTime = 0;
static int autoClusterNumber = 0;
static bool removeEnable = true;

static float lidar_max_z = 3.44f;
static float lidar_min_z = -1.72f+0.2f;

static std::fstream outputFile;

static std::fstream timeFile;

// set initial guess
void transformAssociateToMap()
{
  q_w_curr = q_wmap_wodom * q_wodom_curr;
  t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}

void transformUpdate()
{
  q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
  t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
}

void pointAssociateToMap(PointType const *const pi, PointType *const po)
{
  Eigen::Vector3d point_curr(double(pi->x), double(pi->y), double(pi->z));
  Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
  po->x = float(point_w.x());
  po->y = float(point_w.y());
  po->z = float(point_w.z());
  po->intensity = pi->intensity;
  
  po->normal_x = pi->normal_x;
  po->normal_y = pi->normal_z;
  po->normal_z = pi->normal_z;
  po->curvature = pi->curvature;
  //po->intensity = 1.0;
}

void pointAssociateTobeMapped(PointType const *const pi, PointType *const po)
{
  Eigen::Vector3d point_w(double(pi->x), double(pi->y), double(pi->z));
  Eigen::Vector3d point_curr = q_w_curr.inverse() * (point_w - t_w_curr);
  po->x = float(point_curr.x());
  po->y = float(point_curr.y());
  po->z = float(point_curr.z());
  po->intensity = pi->intensity;
  
  po->normal_x = pi->normal_x;
  po->normal_y = pi->normal_z;
  po->normal_z = pi->normal_z;
  po->curvature = pi->curvature;
}

void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudCornerLast2)
{
  mBuf.lock();
  cornerLastBuf.push(laserCloudCornerLast2);
  mBuf.unlock();
}

void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudSurfLast2)
{
  mBuf.lock();
  surfLastBuf.push(laserCloudSurfLast2);
  mBuf.unlock();
}

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
  mBuf.lock();
  fullResBuf.push(laserCloudFullRes2);
  mBuf.unlock();
}

//receive odomtry
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
{
  mBuf.lock();
  odometryBuf.push(laserOdometry);
  mBuf.unlock();
  
  // high frequence publish
  Eigen::Quaterniond q_wodom_curr;
  Eigen::Vector3d t_wodom_curr;
  q_wodom_curr.x() = laserOdometry->pose.pose.orientation.x;
  q_wodom_curr.y() = laserOdometry->pose.pose.orientation.y;
  q_wodom_curr.z() = laserOdometry->pose.pose.orientation.z;
  q_wodom_curr.w() = laserOdometry->pose.pose.orientation.w;
  t_wodom_curr.x() = laserOdometry->pose.pose.position.x;
  t_wodom_curr.y() = laserOdometry->pose.pose.position.y;
  t_wodom_curr.z() = laserOdometry->pose.pose.position.z;
  
  Eigen::Quaterniond q_w_curr = q_wmap_wodom * q_wodom_curr;
  Eigen::Vector3d t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
  
  nav_msgs::Odometry odomAftMapped;
  odomAftMapped.header.frame_id = "/camera_init";
  odomAftMapped.child_frame_id = "/aft_mapped";
  odomAftMapped.header.stamp = laserOdometry->header.stamp;
  odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
  odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
  odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
  odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
  odomAftMapped.pose.pose.position.x = t_w_curr.x();
  odomAftMapped.pose.pose.position.y = t_w_curr.y();
  odomAftMapped.pose.pose.position.z = t_w_curr.z();
  pubOdomAftMappedHighFrec.publish(odomAftMapped);
}

void process() __attribute__((noreturn));

void process()
{
  while(1) //线程主循环
  {
    while (!cornerLastBuf.empty() && !surfLastBuf.empty() &&
           !fullResBuf.empty() && !odometryBuf.empty())
    {
      mBuf.lock();
      
      ROS_INFO("Buf Size = (%lu,%lu,%lu,%lu)",cornerLastBuf.size(),surfLastBuf.size(),fullResBuf.size(),odometryBuf.size());
      while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
        odometryBuf.pop();
      if (odometryBuf.empty())
      {
        mBuf.unlock();
        break;
      }
      
      while (!surfLastBuf.empty() && surfLastBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
        surfLastBuf.pop();
      if (surfLastBuf.empty())
      {
        mBuf.unlock();
        break;
      }
      
      while (!fullResBuf.empty() && fullResBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
        fullResBuf.pop();
      if (fullResBuf.empty())
      {
        mBuf.unlock();
        break;
      }
      
      timeLaserCloudCornerLast = cornerLastBuf.front()->header.stamp.toSec();
      timeLaserCloudSurfLast = surfLastBuf.front()->header.stamp.toSec();
      timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();
      timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();
      
      if (abs(timeLaserCloudCornerLast - timeLaserOdometry) > 0.01 ||
          abs(timeLaserCloudSurfLast - timeLaserOdometry) > 0.01 ||
          abs(timeLaserCloudFullRes - timeLaserOdometry) > 0.01 )
      {
        printf("time corner %f surf %f full %f odom %f \n", timeLaserCloudCornerLast, timeLaserCloudSurfLast, timeLaserCloudFullRes, timeLaserOdometry);
        printf("unsync messeage!");
        mBuf.unlock();
        break;
      }
      
      laserCloudCornerLast->clear();
      pcl::fromROSMsg(*cornerLastBuf.front(), *laserCloudCornerLast);
      cornerLastBuf.pop();
      
      laserCloudSurfLast->clear();
      pcl::fromROSMsg(*surfLastBuf.front(), *laserCloudSurfLast);
      surfLastBuf.pop();
      
      laserCloudFullRes->clear();
      pcl::fromROSMsg(*fullResBuf.front(), *laserCloudFullRes);
      fullResBuf.pop();
      
      q_wodom_curr.x() = odometryBuf.front()->pose.pose.orientation.x;
      q_wodom_curr.y() = odometryBuf.front()->pose.pose.orientation.y;
      q_wodom_curr.z() = odometryBuf.front()->pose.pose.orientation.z;
      q_wodom_curr.w() = odometryBuf.front()->pose.pose.orientation.w;
      t_wodom_curr.x() = odometryBuf.front()->pose.pose.position.x;
      t_wodom_curr.y() = odometryBuf.front()->pose.pose.position.y;
      t_wodom_curr.z() = odometryBuf.front()->pose.pose.position.z;
      odometryBuf.pop();
      
      while(!cornerLastBuf.empty())
      {
        cornerLastBuf.pop();
        printf("drop lidar frame in mapping for real time performance \n");
      }
      
      mBuf.unlock();
      
      TicToc t_whole;
      
      transformAssociateToMap();
      
      TicToc t_shift;
      int centerCubeI = int((t_w_curr.x() + 25.0) / 50.0) + laserCloudCenWidth;
      int centerCubeJ = int((t_w_curr.y() + 25.0) / 50.0) + laserCloudCenHeight;
      int centerCubeK = int((t_w_curr.z() + 25.0) / 50.0) + laserCloudCenDepth;
      
      if (t_w_curr.x() + 25.0 < 0)
        centerCubeI--;
      if (t_w_curr.y() + 25.0 < 0)
        centerCubeJ--;
      if (t_w_curr.z() + 25.0 < 0)
        centerCubeK--;
      
      while (centerCubeI < 2)
      {
        for (int j = 0; j < laserCloudHeight; j++)
        {
          for (int k = 0; k < laserCloudDepth; k++)
          {
            int i = laserCloudWidth - 1;
            pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            for (; i >= 1; i--)
            {
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCornerArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudSurfArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            }
            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeCornerPointer;
            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeSurfPointer;
            laserCloudCubeCornerPointer->clear();
            laserCloudCubeSurfPointer->clear();
          }
        }
        
        centerCubeI++;
        laserCloudCenWidth++;
      }
      
      while (centerCubeI >= laserCloudWidth - 2)
      {
        for (int j = 0; j < laserCloudHeight; j++)
        {
          for (int k = 0; k < laserCloudDepth; k++)
          {
            int i = 0;
            pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            for (; i < laserCloudWidth - 1; i++)
            {
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCornerArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudSurfArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            }
            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeCornerPointer;
            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeSurfPointer;
            laserCloudCubeCornerPointer->clear();
            laserCloudCubeSurfPointer->clear();
          }
        }
        
        centerCubeI--;
        laserCloudCenWidth--;
      }
      
      while (centerCubeJ < 2)
      {
        for (int i = 0; i < laserCloudWidth; i++)
        {
          for (int k = 0; k < laserCloudDepth; k++)
          {
            int j = laserCloudHeight - 1;
            pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            for (; j >= 1; j--)
            {
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCornerArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudSurfArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
            }
            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeCornerPointer;
            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeSurfPointer;
            laserCloudCubeCornerPointer->clear();
            laserCloudCubeSurfPointer->clear();
          }
        }
        
        centerCubeJ++;
        laserCloudCenHeight++;
      }
      
      while (centerCubeJ >= laserCloudHeight - 2)
      {
        for (int i = 0; i < laserCloudWidth; i++)
        {
          for (int k = 0; k < laserCloudDepth; k++)
          {
            int j = 0;
            pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            for (; j < laserCloudHeight - 1; j++)
            {
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCornerArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudSurfArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
            }
            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeCornerPointer;
            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeSurfPointer;
            laserCloudCubeCornerPointer->clear();
            laserCloudCubeSurfPointer->clear();
          }
        }
        
        centerCubeJ--;
        laserCloudCenHeight--;
      }
      
      while (centerCubeK < 2)
      {
        for (int i = 0; i < laserCloudWidth; i++)
        {
          for (int j = 0; j < laserCloudHeight; j++)
          {
            int k = laserCloudDepth - 1;
            pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            for (; k >= 1; k--)
            {
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
            }
            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeCornerPointer;
            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeSurfPointer;
            laserCloudCubeCornerPointer->clear();
            laserCloudCubeSurfPointer->clear();
          }
        }
        
        centerCubeK++;
        laserCloudCenDepth++;
      }
      
      while (centerCubeK >= laserCloudDepth - 2)
      {
        for (int i = 0; i < laserCloudWidth; i++)
        {
          for (int j = 0; j < laserCloudHeight; j++)
          {
            int k = 0;
            pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            for (; k < laserCloudDepth - 1; k++)
            {
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
            }
            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeCornerPointer;
            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeSurfPointer;
            laserCloudCubeCornerPointer->clear();
            laserCloudCubeSurfPointer->clear();
          }
        }
        
        centerCubeK--;
        laserCloudCenDepth--;
      }
      
      int laserCloudValidNum = 0;
      int laserCloudSurroundNum = 0;
      
      for (int i = centerCubeI - 1; i <= centerCubeI + 1; i++)
      {
        for (int j = centerCubeJ - 1; j <= centerCubeJ + 1; j++)
        {
          for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++)
          {
            if (i >= 0 && i < laserCloudWidth &&
                j >= 0 && j < laserCloudHeight &&
                k >= 0 && k < laserCloudDepth)
            {
              laserCloudValidInd[laserCloudValidNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
              laserCloudValidNum++;
              laserCloudSurroundInd[laserCloudSurroundNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
              laserCloudSurroundNum++;
            }
          }
        }
      }
      
      laserCloudCornerFromMap->clear();
      laserCloudSurfFromMap->clear();
      for (int i = 0; i < laserCloudValidNum; i++)
      {
        *laserCloudCornerFromMap += *laserCloudCornerArray[laserCloudValidInd[i]];
        *laserCloudSurfFromMap += *laserCloudSurfArray[laserCloudValidInd[i]];
      }
      ulong laserCloudCornerFromMapNum = laserCloudCornerFromMap->points.size();
      ulong laserCloudSurfFromMapNum = laserCloudSurfFromMap->points.size();
      
      
      pcl::PointCloud<PointType>::Ptr laserCloudCornerStack(new pcl::PointCloud<PointType>());
      downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
      downSizeFilterCorner.filter(*laserCloudCornerStack);
      ulong laserCloudCornerStackNum = laserCloudCornerStack->points.size();
      
      pcl::PointCloud<PointType>::Ptr laserCloudSurfStack(new pcl::PointCloud<PointType>());
      downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
      downSizeFilterSurf.filter(*laserCloudSurfStack);
      ulong laserCloudSurfStackNum = laserCloudSurfStack->points.size();
      
      printf("map prepare time %f ms\n", t_shift.toc());
      printf("map corner num %lu  surf num %lu \n", laserCloudCornerFromMapNum, laserCloudSurfFromMapNum);
      
      Accumulator<float> groundLevelMean; //地面高度累加器
      
      if (laserCloudCornerFromMapNum > 10 && laserCloudSurfFromMapNum > 50)
      {
        TicToc t_opt;
        TicToc t_tree;
        kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMap);
        kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMap);
        printf("build tree time %f ms \n", t_tree.toc());
        
        for (int iterCount = 0; iterCount < 2; iterCount++)
        {
          //ceres::LossFunction *loss_function = NULL;
          ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
          ceres::LocalParameterization *q_parameterization =
              new ceres::EigenQuaternionParameterization();
          ceres::Problem::Options problem_options;
          
          ceres::Problem problem(problem_options);
          problem.AddParameterBlock(parameters, 4, q_parameterization);
          problem.AddParameterBlock(parameters + 4, 3);
          
          TicToc t_data;
          int corner_num = 0;
          
          for (uint i = 0; i < laserCloudCornerStackNum; i++)
          {
            pointOri = laserCloudCornerStack->points[i];
            groundLevelMean.addDateValue(pointOri.z);
            //double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
            
            // 计算原始点的光流矢量和动态概率
            float opticalX=0.f,opticalY=0.f,opticalZ=0.f,opticalDX=0.f,opticalDY=0.f,opticalDZ=0.f,distance;
            for (uint j = 0; j < 5; ++j) {
              //              opticalX += laserCloudCornerFromMap->points[uint(pointSearchInd[j])].normal_x;
              //              opticalY += laserCloudCornerFromMap->points[uint(pointSearchInd[j])].normal_y;
              //              opticalZ += laserCloudCornerFromMap->points[uint(pointSearchInd[j])].normal_z;
              
              opticalDX += pointSel.x - laserCloudCornerFromMap->points[uint(pointSearchInd[j])].x;
              opticalDY += pointSel.y - laserCloudCornerFromMap->points[uint(pointSearchInd[j])].y;
              opticalDZ += pointSel.z - laserCloudCornerFromMap->points[uint(pointSearchInd[j])].z;
            }
            opticalX = opticalX/5 + opticalDX/5;
            opticalY = opticalY/5 + opticalDY/5;
            opticalZ = opticalZ/5 + opticalDZ/5;
            
            laserCloudCornerStack->points[i].normal_x = opticalX;
            laserCloudCornerStack->points[i].normal_y = opticalY;
            laserCloudCornerStack->points[i].normal_z = opticalZ;
            distance = sqrt(opticalX*opticalX+opticalY*opticalY+opticalZ*opticalZ);
            
            connerOpticalDistanceMean.addDateValue(distance);
            
            
            if (pointSearchSqDis[4] < 1.0f)
            {
              std::vector<Eigen::Vector3d> nearCorners;
              Eigen::Vector3d center(0, 0, 0);
              for (uint j = 0; j < 5; j++)
              {
                Eigen::Vector3d tmp(double(laserCloudCornerFromMap->points[uint(pointSearchInd[j])].x),
                    double(laserCloudCornerFromMap->points[uint(pointSearchInd[j])].y),
                    double(laserCloudCornerFromMap->points[uint(pointSearchInd[j])].z));
                center = center + tmp;
                nearCorners.push_back(tmp);
              }
              center = center / 5.0;
              
              Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
              for (uint j = 0; j < 5; j++)
              {
                Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
              }
              
              Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
              
              // if is indeed line feature
              // note Eigen library sort eigenvalues in increasing order
              Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
              Eigen::Vector3d curr_point(double(pointOri.x), double(pointOri.y), double(pointOri.z));
              if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
              {
                Eigen::Vector3d point_on_line = center;
                Eigen::Vector3d point_a, point_b;
                point_a = 0.1 * unit_direction + point_on_line;
                point_b = -0.1 * unit_direction + point_on_line;
                
                ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
                problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
                corner_num++;
              }
            }
            /*
            else if(pointSearchSqDis[4] < 0.01 * sqrtDis)
            {
              Eigen::Vector3d center(0, 0, 0);
              for (int j = 0; j < 5; j++)
              {
                Eigen::Vector3d tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
                          laserCloudCornerFromMap->points[pointSearchInd[j]].y,
                          laserCloudCornerFromMap->points[pointSearchInd[j]].z);
                center = center + tmp;
              }
              center = center / 5.0;
              Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
              ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, center);
              problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
            }
            */
          }
          
          int surf_num = 0;
          for (uint i = 0; i < laserCloudSurfStackNum; i++)
          {
            pointOri = laserCloudSurfStack->points[i];
            groundLevelMean.addDateValue(pointOri.z);
            //double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
            
            // 计算原始点的光流矢量
            float opticalX=0.f,opticalY=0.f,opticalZ=0.f,opticalDX=0.f,opticalDY=0.f,opticalDZ=0.f,distance;
            for (uint j = 0; j < 5; ++j) {
              //              opticalX += laserCloudSurfFromMap->points[pointSearchInd[j]].normal_x;
              //              opticalY += laserCloudSurfFromMap->points[pointSearchInd[j]].normal_y;
              //              opticalZ += laserCloudSurfFromMap->points[pointSearchInd[j]].normal_z;
              
              opticalDX += pointSel.x - laserCloudSurfFromMap->points[uint(pointSearchInd[j])].x;
              opticalDY += pointSel.y - laserCloudSurfFromMap->points[uint(pointSearchInd[j])].y;
              opticalDZ += pointSel.z - laserCloudSurfFromMap->points[uint(pointSearchInd[j])].z;
            }
            
            opticalX = opticalX/5 + opticalDX/5;
            opticalY = opticalY/5 + opticalDY/5;
            opticalZ = opticalZ/5 + opticalDZ/5;
            
            laserCloudSurfStack->points[i].normal_x = opticalX;
            laserCloudSurfStack->points[i].normal_y = opticalY;
            laserCloudSurfStack->points[i].normal_z = opticalZ;
            distance = sqrt(opticalX*opticalX+opticalY*opticalY+opticalZ*opticalZ);
            
            surfOpticalDistanceMean.addDateValue(distance);
            
            Eigen::Matrix<double, 5, 3> matA0;
            Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
            if (pointSearchSqDis[4] < 1.0f)
            {
              
              for (uint j = 0; j < 5; j++)
              {
                matA0(j, 0) = double(laserCloudSurfFromMap->points[uint(pointSearchInd[j])].x);
                matA0(j, 1) = double(laserCloudSurfFromMap->points[uint(pointSearchInd[j])].y);
                matA0(j, 2) = double(laserCloudSurfFromMap->points[uint(pointSearchInd[j])].z);
                //printf(" pts %f %f %f ", matA0(j, 0), matA0(j, 1), matA0(j, 2));
              }
              // find the norm of plane
              Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
              double negative_OA_dot_norm = 1 / norm.norm();
              norm.normalize();
              
              // Here n(pa, pb, pc) is unit norm of plane
              bool planeValid = true;
              for (uint j = 0; j < 5; j++)
              {
                // if OX * n > 0.2, then plane is not fit well
                if (fabs(norm(0) * double(laserCloudSurfFromMap->points[uint(pointSearchInd[j])].x) +
                         norm(1) * double(laserCloudSurfFromMap->points[uint(pointSearchInd[j])].y) +
                         norm(2) * double(laserCloudSurfFromMap->points[uint(pointSearchInd[j])].z) + negative_OA_dot_norm) > 0.2)
                {
                  planeValid = false;
                  break;
                }
              }
              Eigen::Vector3d curr_point(double(pointOri.x), double(pointOri.y), double(pointOri.z));
              if (planeValid)
              {
                ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, norm, negative_OA_dot_norm);
                problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
                surf_num++;
              }
            }
            /*
            else if(pointSearchSqDis[4] < 0.01 * sqrtDis)
            {
              Eigen::Vector3d center(0, 0, 0);
              for (int j = 0; j < 5; j++)
              {
                Eigen::Vector3d tmp(laserCloudSurfFromMap->points[pointSearchInd[j]].x,
                          laserCloudSurfFromMap->points[pointSearchInd[j]].y,
                          laserCloudSurfFromMap->points[pointSearchInd[j]].z);
                center = center + tmp;
              }
              center = center / 5.0;
              Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
              ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, center);
              problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
            }
            */
          }
          
          //printf("corner num %d used corner num %d \n", laserCloudCornerStackNum, corner_num);
          //printf("surf num %d used surf num %d \n", laserCloudSurfStackNum, surf_num);
          
          printf("mapping data assosiation time %f ms \n", t_data.toc());
          printf("connerOpticalDistanceMean is %f m with %f m. \n", double(connerOpticalDistanceMean.mean()), double(connerOpticalDistanceMean.stddev()));
          printf("surfOpticalDistanceMean is %f m with %f m. \n", double(surfOpticalDistanceMean.mean()), double(surfOpticalDistanceMean.stddev()));
          
          TicToc t_solver;
          ceres::Solver::Options options;
          options.linear_solver_type = ceres::DENSE_QR;
          options.max_num_iterations = 4;
          options.minimizer_progress_to_stdout = false;
          options.check_gradients = false;
          options.gradient_check_relative_precision = 1e-4;
          ceres::Solver::Summary summary;
          ceres::Solve(options, &problem, &summary);
          printf("mapping solver time %f ms \n", t_solver.toc());
          
          //printf("time %f \n", timeLaserOdometry);
          //printf("corner factor num %d surf factor num %d\n", corner_num, surf_num);
          //printf("result q %f %f %f %f result t %f %f %f\n", parameters[3], parameters[0], parameters[1], parameters[2],
          //	   parameters[4], parameters[5], parameters[6]);
        }
        printf("mapping optimization time %f \n", t_opt.toc());
      }
      else
      {
        ROS_WARN("time Map corner and surf num are not enough");
      }
      transformUpdate();


      TicToc t_remove;
      // 计算类别运动概率
      pcl::PointCloud<PointType>::Ptr laserCloudStack2(new pcl::PointCloud<PointType>());

      *laserCloudStack2 += *laserCloudCornerStack;
      *laserCloudStack2 += *laserCloudSurfStack;

      pcl::PointCloud<PointType>::Ptr laserCloudStackWithoutGroud2(new pcl::PointCloud<PointType>());
      pcl::PassThrough<PointType> pass;
      pass.setInputCloud(laserCloudStack2);
      pass.setFilterFieldName("z");

      groundLevelMean.mean();
      pass.setFilterLimits((lidar_min_z+groundLevelMean.mean())/2.0,lidar_max_z);
      pass.filter(*laserCloudStackWithoutGroud2);

      pcl::PointCloud<PointType>::Ptr laserCloudStack(new pcl::PointCloud<PointType>());
      for (uint i = 0; i < laserCloudStack2->size(); i++)
      {
        pointAssociateToMap(&laserCloudStack2->points[i], &pointSel);
        pointSel.intensity = laserCloudStack2->points[i].z;
        laserCloudStack->push_back(pointSel);
      }

      pcl::PointCloud<PointType>::Ptr laserCloudStackWithoutGroud(new pcl::PointCloud<PointType>());
      for (uint i = 0; i < laserCloudStackWithoutGroud2->size(); i++)
      {
        pointAssociateToMap(&laserCloudStackWithoutGroud2->points[i], &pointSel);
        pointSel.intensity = laserCloudStack2->points[i].z;
        laserCloudStackWithoutGroud->push_back(pointSel);
      }

      Eigen::Array3f min_size,max_size;
      min_size.x() = cluster_min_size;
      min_size.y() = cluster_min_size;
      min_size.z() = cluster_min_size;
      max_size.x() = cluster_max_size;
      max_size.y() = cluster_max_size;
      max_size.z() = cluster_max_size;

      //聚类
      hdl_people_detection::MarcelPeopleDetector marcel(min_pts, max_pts, min_size, max_size,cluster_tolerane,cluster_lambda);
      auto clusters = marcel.detect(laserCloudStackWithoutGroud);

      //计算类别运动距离
      ulong clusterSize = clusters.size();
      std::vector<float> dist(clusterSize);
      for (uint i = 0; i < clusterSize; i++) {
        Accumulator<float> normal[3];
        for (auto &pt: clusters[i]->cloud->points) {
          normal[0].addDateValue(pt.normal_x);
          normal[1].addDateValue(pt.normal_y);
          normal[2].addDateValue(pt.normal_z);
        }

        PointType pt_map,pt_laser;
        pt_map.x = clusters[i]->centroid.x();
        pt_map.y = clusters[i]->centroid.y();
        pt_map.z = clusters[i]->centroid.z();
        pointAssociateTobeMapped(&pt_map,&pt_laser);

        dist[i] = sqrt(normal[0].mean()*normal[0].mean()+normal[1].mean()*normal[1].mean()+normal[2].mean()*normal[2].mean())/sqrt(pt_laser.x*pt_laser.x+pt_laser.y*pt_laser.y+pt_laser.z*pt_laser.z);

        objectSpeedMean.addDateValue(dist[i]);
      }

      aloam_velodyne::ClusterArray::Ptr clusters_msg(new aloam_velodyne::ClusterArray());
      clusters_msg->header.frame_id = "/aft_mapped";
      clusters_msg->header.stamp = ros::Time().fromSec(timeLaserOdometry);

      for(uint i=0; i<clusters.size(); i++) {
        aloam_velodyne::Cluster cluster_msg;
        cluster_msg.id = i;
        cluster_msg.is_human = clusters[i]->is_human;
        cluster_msg.min_pt.x = double(clusters[i]->min_pt.x());
        cluster_msg.min_pt.y = double(clusters[i]->min_pt.y());
        cluster_msg.min_pt.z = double(clusters[i]->min_pt.z());

        cluster_msg.max_pt.x = double(clusters[i]->max_pt.x());
        cluster_msg.max_pt.y = double(clusters[i]->max_pt.y());
        cluster_msg.max_pt.z = double(clusters[i]->max_pt.z());

        cluster_msg.size.x = double(clusters[i]->size.x());
        cluster_msg.size.y = double(clusters[i]->size.y());
        cluster_msg.size.z = double(clusters[i]->size.z());

        cluster_msg.centroid.x = double(clusters[i]->centroid.x());
        cluster_msg.centroid.y = double(clusters[i]->centroid.y());
        cluster_msg.centroid.z = double(clusters[i]->centroid.z());

        clusters_msg->clusters.push_back(cluster_msg);
      }

      tracker->predict(clusters_msg->header.stamp);
      tracker->correct(clusters_msg->header.stamp,clusters_msg->clusters);
      //printf("There are %zu dynamic objects.\n", tracker->people.size());

      //计算每个跟踪类的速度
      auto associations = tracker->data_association->associate(tracker->people,clusters_msg->clusters);

      std::vector<double> speed(associations.size());
      for (uint i=0; i<associations.size(); i++) {
        auto people = tracker->people[uint(associations[i].tracker)];
        auto cluster = clusters_msg->clusters[uint(associations[i].observation)];

        PointType pt_map,pt_laser;
        pt_map.x = cluster.centroid.x;
        pt_map.y = cluster.centroid.y;
        pt_map.z = cluster.centroid.z;
        pointAssociateTobeMapped(&pt_map,&pt_laser);

        speed[i] = people->velocity().norm()/sqrt(pt_laser.x*pt_laser.x+pt_laser.y*pt_laser.y+pt_laser.z*pt_laser.z);

        tarckerSpeedMean.addDateValue(speed[i]);

        for (auto &pt: clusters[cluster.id]->cloud->points) {
          pt.intensity = float(dist[clusters_msg->clusters[uint(associations[i].observation)].id]);
          //pt.intensity = float(speed[i]);
        }
        *currDynamicCloudStack += *clusters[cluster.id]->cloud;
      }

      printf("There are %zu clusters, and moved %f m with %f m.\n", dist.size(), double(objectSpeedMean.mean()), double(objectSpeedMean.stddev()));
      printf("There are %zu trackers, and moved %f m/s with %f m/s.\n", associations.size(), tarckerSpeedMean.mean(), tarckerSpeedMean.stddev());

      maxObjectSpeed = tarckerSpeedMean.mean()/2.0f;
      maxClassDist = objectSpeedMean.mean()/2.0f;

      //移除跟踪类型的
      for (uint i=0; i<associations.size(); i++) {
        if(speed[i]>maxObjectSpeed||dist[clusters_msg->clusters[uint(associations[i].observation)].id]>maxClassDist)
        {
          auto cluster = clusters_msg->clusters[uint(associations[i].observation)];

          // build the condition
          pcl::ConditionOr<PointType>::Ptr range_cond (new pcl::ConditionOr<PointType> ());
          range_cond->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("x", pcl::ComparisonOps::LT, cluster.min_pt.x-groundLevelMean.stddev())));
          range_cond->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("x", pcl::ComparisonOps::GT, cluster.max_pt.x+groundLevelMean.stddev())));
          range_cond->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("y", pcl::ComparisonOps::LT, cluster.min_pt.y-groundLevelMean.stddev())));
          range_cond->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("y", pcl::ComparisonOps::GT, cluster.max_pt.y+groundLevelMean.stddev())));
          range_cond->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("z", pcl::ComparisonOps::LT, cluster.min_pt.z-groundLevelMean.stddev())));
          range_cond->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("z", pcl::ComparisonOps::GT, cluster.max_pt.z+groundLevelMean.stddev())));

          // build the filter
          pcl::ConditionalRemoval<PointType> condrem;
          condrem.setCondition (range_cond);
          condrem.setInputCloud (laserCloudStack);
          condrem.setKeepOrganized(false);

          // apply filter
          if(removeEnable)
          {
            condrem.filter (*laserCloudStack);
          }

          //printf("No. %d speed = %f, dist = %f\n", clusters_msg->clusters[uint(associations[i].observation)].id, speed[i], dist[clusters_msg->clusters[uint(associations[i].observation)].id]);
        }
      }

      *staticCloudMap += *laserCloudStack;
      downSizeFilterCorner.setInputCloud(staticCloudMap);
      downSizeFilterCorner.filter(*staticCloudMap);

      pcl::ConditionOr<PointType>::Ptr corner_cond (new pcl::ConditionOr<PointType> ());
      corner_cond->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("curvature", pcl::ComparisonOps::LT, 0.1)));
      pcl::ConditionOr<PointType>::Ptr surf_cond (new pcl::ConditionOr<PointType> ());
      surf_cond->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new pcl::FieldComparison<PointType> ("curvature", pcl::ComparisonOps::GT, 0.1)));
      // build the filter
      pcl::ConditionalRemoval<PointType> condrem;
      condrem.setCondition (corner_cond);
      condrem.setInputCloud (laserCloudStack);
      condrem.setKeepOrganized(false);

      laserCloudCornerStack->clear();
      condrem.filter (*laserCloudCornerStack);

      condrem.setCondition (surf_cond);
      condrem.setInputCloud (laserCloudStack);
      condrem.setKeepOrganized(false);

      laserCloudSurfStack->clear();
      condrem.filter (*laserCloudSurfStack);

      double remove_time = t_remove.toc();
      removeTimeMean.addDateValue(remove_time);
      timeFile<<remove_time<<std::endl;

      TicToc t_add;

      ulong laserCloudCornerStackNum2 = laserCloudCornerStack->points.size();
      ulong laserCloudSurfStackNum2 = laserCloudSurfStack->points.size();

      for (uint i = 0; i < laserCloudCornerStackNum2; i++)
      {
        //pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel);
        pointSel = laserCloudCornerStack->points[i];
        
        int cubeI = int((pointSel.x + 25.0f) / 50.0f) + laserCloudCenWidth;
        int cubeJ = int((pointSel.y + 25.0f) / 50.0f) + laserCloudCenHeight;
        int cubeK = int((pointSel.z + 25.0f) / 50.0f) + laserCloudCenDepth;
        
        if (pointSel.x + 25.0f < 0.f)
          cubeI--;
        if (pointSel.y + 25.0f < 0.f)
          cubeJ--;
        if (pointSel.z + 25.0f < 0.f)
          cubeK--;
        
        if (cubeI >= 0 && cubeI < laserCloudWidth &&
            cubeJ >= 0 && cubeJ < laserCloudHeight &&
            cubeK >= 0 && cubeK < laserCloudDepth)
        {
          int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
          laserCloudCornerArray[cubeInd]->push_back(pointSel);
        }
      }
      
      for (uint i = 0; i < laserCloudSurfStackNum2; i++)
      {
        //pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel);
        pointSel = laserCloudSurfStack->points[i];
        
        int cubeI = int((pointSel.x + 25.0f) / 50.0f) + laserCloudCenWidth;
        int cubeJ = int((pointSel.y + 25.0f) / 50.0f) + laserCloudCenHeight;
        int cubeK = int((pointSel.z + 25.0f) / 50.0f) + laserCloudCenDepth;
        
        if (pointSel.x + 25.0f < 0.f)
          cubeI--;
        if (pointSel.y + 25.0f < 0.f)
          cubeJ--;
        if (pointSel.z + 25.0f < 0.f)
          cubeK--;
        
        if (cubeI >= 0 && cubeI < laserCloudWidth &&
            cubeJ >= 0 && cubeJ < laserCloudHeight &&
            cubeK >= 0 && cubeK < laserCloudDepth)
        {
          int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
          laserCloudSurfArray[cubeInd]->push_back(pointSel);
        }
      }
      printf("add points time %f ms\n", t_add.toc());
      
      TicToc t_filter;
      for (int i = 0; i < laserCloudValidNum; i++)
      {
        int ind = laserCloudValidInd[i];
        
        pcl::PointCloud<PointType>::Ptr tmpCorner(new pcl::PointCloud<PointType>());
        downSizeFilterCorner.setInputCloud(laserCloudCornerArray[ind]);
        downSizeFilterCorner.filter(*tmpCorner);
        laserCloudCornerArray[ind] = tmpCorner;
        
        pcl::PointCloud<PointType>::Ptr tmpSurf(new pcl::PointCloud<PointType>());
        downSizeFilterSurf.setInputCloud(laserCloudSurfArray[ind]);
        downSizeFilterSurf.filter(*tmpSurf);
        laserCloudSurfArray[ind] = tmpSurf;
      }
      printf("filter time %f ms \n", t_filter.toc());
      
      TicToc t_pub;
      //publish surround map for every 5 frame
      if (frameCount % 5 == 0)
      {
        laserCloudSurround->clear();
        for (int i = 0; i < laserCloudSurroundNum; i++)
        {
          int ind = laserCloudSurroundInd[i];
          *laserCloudSurround += *laserCloudCornerArray[ind];
          *laserCloudSurround += *laserCloudSurfArray[ind];
        }
        
        sensor_msgs::PointCloud2 laserCloudSurround3;
        pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
        laserCloudSurround3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        laserCloudSurround3.header.frame_id = "/camera_init";
        pubLaserCloudSurround.publish(laserCloudSurround3);
      }
      
      if (frameCount % 20 == 0)
      {
        pcl::PointCloud<PointType> laserCloudMap;
        for (int i = 0; i < 4851; i++)
        {
          laserCloudMap += *laserCloudCornerArray[i];
          laserCloudMap += *laserCloudSurfArray[i];
        }
        sensor_msgs::PointCloud2 laserCloudMsg;
        pcl::toROSMsg(laserCloudMap, laserCloudMsg);
        laserCloudMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        laserCloudMsg.header.frame_id = "/camera_init";
        pubLaserCloudMap.publish(laserCloudMsg);
        
        pcl::toROSMsg(*staticCloudMap, laserCloudMsg);
        laserCloudMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        laserCloudMsg.header.frame_id = "/camera_init";
        pubStaticCloudMap.publish(laserCloudMsg);
      }
      
      ulong laserCloudFullResNum = laserCloudFullRes->points.size();
      for (uint i = 0; i < laserCloudFullResNum; i++)
      {
        pointAssociateToMap(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
      }
      
      sensor_msgs::PointCloud2 laserCloudFullRes3;
      pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
      laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
      laserCloudFullRes3.header.frame_id = "/camera_init";
      pubLaserCloudFullRes.publish(laserCloudFullRes3);
      
      sensor_msgs::PointCloud2 laserCloudFullRes4;
      pcl::toROSMsg(*currDynamicCloudStack, laserCloudFullRes4);
      laserCloudFullRes4.header.stamp = ros::Time().fromSec(timeLaserOdometry);
      laserCloudFullRes4.header.frame_id = "/camera_init";
      pubDynamicCloudCurr.publish(laserCloudFullRes4);
      
      printf("mapping pub time %f ms \n", t_pub.toc());
      
      double mapping_whole = t_whole.toc();
      printf("whole mapping time %f ms +++++\n", mapping_whole);
      
      if(autoMapping)
      {
        float rate_conner = 0.1f*laserCloudCornerStackNum/(laserCloudCornerStackNum+laserCloudSurfStackNum);
        float rate_surf = 0.1f - rate_conner;
        if(mapping_whole>autoMappingTime*1.1)
        {
          lineRes = lineRes*(1+rate_conner);
          planeRes = planeRes*(1+rate_surf);
        }
        else if (mapping_whole<autoMappingTime*0.9)
        {
          lineRes = lineRes*(1-rate_conner);
          planeRes = planeRes*(1-rate_surf);
          lineRes = lineRes>0.02f?lineRes:0.02f;
          planeRes = planeRes>0.02f?planeRes:0.02f;
        }
        downSizeFilterCorner.setLeafSize(lineRes, lineRes,lineRes);
        downSizeFilterSurf.setLeafSize(planeRes, planeRes, planeRes);
        printf("line resolution %f plane resolution %f \n", double(lineRes), double(planeRes));
        cluster_tolerane = 2.0f*(lineRes+planeRes);
        cluster_lambda = 4.0f*(lineRes+planeRes);
      }
      nav_msgs::Odometry odomAftMapped;
      odomAftMapped.header.frame_id = "/camera_init";
      odomAftMapped.child_frame_id = "/aft_mapped";
      odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
      odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
      odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
      odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
      odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
      odomAftMapped.pose.pose.position.x = t_w_curr.x();
      odomAftMapped.pose.pose.position.y = t_w_curr.y();
      odomAftMapped.pose.pose.position.z = t_w_curr.z();
      pubOdomAftMapped.publish(odomAftMapped);
      
      
      geometry_msgs::PoseStamped laserAfterMappedPose;
      laserAfterMappedPose.header = odomAftMapped.header;
      laserAfterMappedPose.pose = odomAftMapped.pose.pose;
      laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;
      laserAfterMappedPath.header.frame_id = "/camera_init";
      laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
      pubLaserAfterMappedPath.publish(laserAfterMappedPath);
      
      static tf::TransformBroadcaster br;
      tf::Transform transform;
      tf::Quaternion q;
      transform.setOrigin(tf::Vector3(t_w_curr(0),
                                      t_w_curr(1),
                                      t_w_curr(2)));
      q.setW(q_w_curr.w());
      q.setX(q_w_curr.x());
      q.setY(q_w_curr.y());
      q.setZ(q_w_curr.z());
      transform.setRotation(q);
      br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "/camera_init", "/aft_mapped"));
      
      Eigen::Vector3d tanslate = Eigen::Vector3d(-t_w_curr.y(),-t_w_curr.z(),t_w_curr.x());
      Eigen::Quaterniond rotate = Eigen::Quaterniond(q_w_curr.w(),-q_w_curr.y(),-q_w_curr.z(),q_w_curr.x());
      Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
      T.rotate(rotate);
      T.pretranslate(tanslate);
      //      Eigen::Vector3d ea(-M_PI_2,M_PI_2,0);
      //      T.rotate(Eigen::AngleAxisd(ea[0], Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(ea[1], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(ea[2], Eigen::Vector3d::UnitX()));
      
      Eigen::Matrix4d M = T.matrix();
      
      outputFile<<M(0,0)<<" "<<M(0,1)<<" "<<M(0,2)<<" "<<M(0,3)<<" "<<M(1,0)<<" "<<M(1,1)<<" "<<M(1,2)<<" "<<M(1,3)<<" "<<M(2,0)<<" "<<M(2,1)<<" "<<M(2,2)<<" "<<M(2,3)<<std::endl;
      
      frameCount++;
    }
    std::chrono::milliseconds dura(2);
    std::this_thread::sleep_for(dura);
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  
  nh.param<float>("mapping_line_resolution", lineRes, 0.1f);
  nh.param<float>("mapping_plane_resolution", planeRes, 0.1f);
  printf("line resolution %f plane resolution %f \n", double(lineRes), double(planeRes));
  nh.param<float>("max_object_speed", maxObjectSpeed, 0.2f);
  nh.param<float>("max_class_dist", maxClassDist, 0.1f);
  printf("max_object_speed %f, max_class_dist %f \n", double(maxObjectSpeed), double(maxClassDist));
  
  nh.param<int>("cluster_min_pts", min_pts, 8);
  nh.param<int>("cluster_max_pts", max_pts, 8192);
  nh.param<float>("cluster_min_size", cluster_min_size, 0.2f);
  nh.param<float>("cluster_max_size", cluster_max_size, 20.0);
  nh.param<float>("cluster_tolerane", cluster_tolerane, 0.2f);
  nh.param<float>("cluster_lambda", cluster_lambda, 20.0);
  printf("cluster_pts [%d,%d], cluster_min_size [%f,%f]\n", min_pts, max_pts, double(cluster_min_size), double(cluster_max_size));

  nh.param<bool>("auto_mapping", autoMapping, true);
  nh.param<float>("auto_mapping_time", autoMappingTime, 400.0);
  nh.param<int>("auto_cluster_number", autoClusterNumber, 100);
  nh.param<bool>("remove_enable", removeEnable, true);
  printf("auto_mapping_time %f ms, auto_cluster_number %d \n", double(autoMappingTime), autoClusterNumber);
  
  nh.param<float>("lidar_min_z", lidar_min_z, -1.73f);
  nh.param<float>("lidar_max_z", lidar_max_z, 3.44f);
  printf("lidar_limit_z ( %f , %f ) m\n", double(lidar_min_z), double(lidar_max_z));
  
  downSizeFilterCorner.setLeafSize(lineRes, lineRes,lineRes);
  downSizeFilterSurf.setLeafSize(planeRes, planeRes, planeRes);
  
  ros::Subscriber subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100, laserCloudCornerLastHandler);
  
  ros::Subscriber subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100, laserCloudSurfLastHandler);
  
  ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 100, laserOdometryHandler);
  
  ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100, laserCloudFullResHandler);
  
  pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 100);
  
  pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 100);
  
  pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100);
  
  pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 100);
  
  pubOdomAftMappedHighFrec = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 100);
  
  pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 100);
  
  pubStaticCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/static_cloud_map", 100);
  
  pubDynamicCloudCurr = nh.advertise<sensor_msgs::PointCloud2>("/dynamic_cloud_curr", 100);
  
  for (int i = 0; i < laserCloudNum; i++)
  {
    laserCloudCornerArray[i].reset(new pcl::PointCloud<PointType>());
    laserCloudSurfArray[i].reset(new pcl::PointCloud<PointType>());
  }
  
  connerOpticalDistanceMean.addDateValue(1.f);
  surfOpticalDistanceMean.addDateValue(.2f);
  objectSpeedMean.addDateValue(.3f);
  tarckerSpeedMean.addDateValue(0.2);
  
  tracker.reset(new hdl_people_tracking::PeopleTracker(pnh));
  
  std::thread mapping_process{process};
  
  outputFile.open("/home/tyin/output.txt", std::ios::out | std::ios::trunc );
  timeFile.open("/home/tyin/time.txt", std::ios::out | std::ios::trunc );
  
  ros::spin();
  
  outputFile.close();
  timeFile.close();
  
  printf("mean %f , stddev %f , min %f , max %f .\n",removeTimeMean.mean(),removeTimeMean.stddev(),removeTimeMean.min(),removeTimeMean.max());
  
  return 0;
}
