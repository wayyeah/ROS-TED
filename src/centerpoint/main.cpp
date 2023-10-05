/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <cmath>
#include <numeric>
#include<time.h> 
#include "paddle/include/paddle_inference_api.h"
#include <dlfcn.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h> 
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <memory>
#include <chrono>
#include <dirent.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <map>
#include <algorithm>
#include <cassert>
#include <unistd.h>
#include <string>
#include <cstdio>
#include<thread>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <tf/transform_datatypes.h>
#include<time.h> 

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}
using paddle_infer::Config;
using paddle_infer::CreatePredictor;
using paddle_infer::Predictor;
std::shared_ptr<paddle_infer::Predictor> create_predictor(
    const std::string &model_path, const std::string &params_path,
    const int gpu_id, const int use_trt, const int trt_precision,
    const int trt_use_static, const std::string trt_static_dir,
    const int collect_shape_info, const std::string dynamic_shape_file) {
  paddle::AnalysisConfig config;
  config.EnableUseGpu(1000, gpu_id);
  config.SetModel(model_path, params_path);
  if (use_trt) {
    paddle::AnalysisConfig::Precision precision;
    if (trt_precision == 0) {
      precision = paddle_infer::PrecisionType::kFloat32;
    } else if (trt_precision == 1) {
      precision = paddle_infer::PrecisionType::kHalf;
    } else {
      LOG(ERROR) << "Tensorrt type can only support 0 or 1, but recieved is"
                 << trt_precision << "\n";
      return nullptr;
    }
    config.EnableTensorRtEngine(1 << 30, 1, 40, precision, trt_use_static,
                                false);

    if (dynamic_shape_file == "") {
      LOG(ERROR) << "dynamic_shape_file should be set, but recieved is "
                 << dynamic_shape_file << "\n";
      return nullptr;
    }
    if (collect_shape_info) {
      config.CollectShapeRangeInfo(dynamic_shape_file);
    } else {
      config.EnableTunedTensorRtDynamicShape(dynamic_shape_file, true);
    }

    if (trt_use_static) {
      if (trt_static_dir == "") {
        LOG(ERROR) << "trt_static_dir should be set, but recieved is "
                   << trt_static_dir << "\n";
        return nullptr;
      }
      config.SetOptimCacheDir(trt_static_dir);
    }
  }
  config.SwitchIrOptim(true);
  return paddle_infer::CreatePredictor(config);
}
void *handle = dlopen("/home/nvidia/way/Ted-Paddle3D/deploy/ted_5.2/cpp/build/libpd_infer_custom_op.so", RTLD_NOW);
auto predictor = create_predictor(
      "/home/nvidia/way/Ted-Paddle3D/output/ted.pdmodel", "/home/nvidia/way/Ted-Paddle3D/output/ted.pdiparams", 0, 0,
     0, 0, " ",
      0, " ");;
std::vector<float> point_cloud_range;


bool read_point(const std::string &file_path, const int num_point_dim,
                void **buffer, int *num_points) {
  std::ifstream file_in(file_path, std::ios::in | std::ios::binary);
  if (num_point_dim < 4) {
    LOG(ERROR) << "Point dimension must not be less than 4, but recieved "
               << "num_point_dim is " << num_point_dim << ".\n";
  }

  if (!file_in) {
    LOG(ERROR) << "Failed to read file: " << file_path << "\n";
    return false;
  }

  std::streampos file_size;
  file_in.seekg(0, std::ios::end);
  file_size = file_in.tellg();
  file_in.seekg(0, std::ios::beg);

  *buffer = malloc(file_size);
  if (*buffer == nullptr) {
    LOG(ERROR) << "Failed to malloc memory of size: " << file_size << "\n";
    return false;
  }
  file_in.read(reinterpret_cast<char *>(*buffer), file_size);
  file_in.close();

  if (file_size / sizeof(float) % num_point_dim != 0) {
    LOG(ERROR) << "Loaded file size (" << file_size
               << ") is not evenly divisible by num_point_dim ("
               << num_point_dim << ")\n";
    return false;
  }
  *num_points = file_size / sizeof(float) / num_point_dim;
  return true;
}

void mask_points_outside_range(const float *points, const int num_points,
                               const std::vector<float> &point_cloud_range,
                               const int num_point_dim,
                               std::vector<float> *selected_points) {
  for (int i = 0; i < num_points; i += num_point_dim) {
    float pt_x = points[i];
    float pt_y = points[i + 1];
    // in [-x, x] and [-y, y] range
    if ((pt_x >= point_cloud_range[0]) && (pt_x <= point_cloud_range[3]) &&
        (pt_y >= point_cloud_range[1]) && (pt_y <= point_cloud_range[4])) {
      for (int d = 0; d < num_point_dim; ++d) {
        selected_points->emplace_back(points[i + d]);
      }
    }
  }
}
bool preprocess(const pcl::PointCloud<pcl::PointXYZ>& cloud, const int num_point_dim,
                const std::vector<float> &point_cloud_range,
                std::vector<int> *points_shape,
                std::vector<float> *points_data) {

    std::vector<float> raw_points;
    raw_points.reserve(cloud.points.size() * num_point_dim);

    for (const auto& point : cloud.points) {
        raw_points.emplace_back(point.x);
        raw_points.emplace_back(point.y);
        raw_points.emplace_back(point.z);
        raw_points.emplace_back(0.0f);  // placeholder for intensity
    }

    std::vector<float> masked_points;
    mask_points_outside_range(raw_points.data(), cloud.points.size(), point_cloud_range,
                            num_point_dim, &masked_points);

    points_data->assign(masked_points.begin(), masked_points.end());
    points_shape->push_back(masked_points.size() / num_point_dim);
    points_shape->push_back(num_point_dim);

    return true;
}
bool preprocess_raw(const std::string &file_path, const int num_point_dim,
                const std::vector<float> &point_cloud_range,
                std::vector<int> *points_shape,
                std::vector<float> *points_data) {
  void *buffer = nullptr;
  int num_points = 0;
  if (!read_point(file_path, num_point_dim, &buffer, &num_points)) {
    return false;
  }
  float *points = static_cast<float *>(buffer);

  std::vector<float> masked_points;
  mask_points_outside_range(points, num_points, point_cloud_range,
                            num_point_dim, &masked_points);

  points_data->assign(masked_points.begin(), masked_points.end());
  points_shape->push_back(masked_points.size() / num_point_dim);
  points_shape->push_back(num_point_dim);

  free(points);
  return true;
}
void run(Predictor *predictor, const std::vector<int> &points_shape,
         const std::vector<float> &points_data, std::vector<float> *box3d_lidar,
         std::vector<int64_t> *label_preds, std::vector<float> *scores) {
  auto input_names = predictor->GetInputNames();
  for (const auto &tensor_name : input_names) {
    auto in_tensor = predictor->GetInputHandle(tensor_name);
    if (tensor_name == "data") {
      in_tensor->Reshape(points_shape);
      in_tensor->CopyFromCpu(points_data.data());
    }
  }

  CHECK(predictor->Run());
  auto output_names = predictor->GetOutputNames();
  for (size_t i = 0; i != output_names.size(); i++) {
    auto output = predictor->GetOutputHandle(output_names[i]);
    std::vector<int> output_shape = output->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                  std::multiplies<int>());
    if (i == 0) {
      box3d_lidar->resize(out_num);
      output->CopyToCpu(box3d_lidar->data());
    } else if (i == 1) {
      scores->resize(out_num);
      output->CopyToCpu(scores->data());
    } else if (i == 2) {
      label_preds->resize(out_num);
      output->CopyToCpu(label_preds->data());
    }
  }
}



ros::Publisher pub;

bool fileExists( std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}



class RosSub_Pub
{  
public:  
    RosSub_Pub() 
    {        
        pub_1 = n.advertise<sensor_msgs::PointCloud2>("/my_topic", 10, true);
        sub_1 = n.subscribe<sensor_msgs::PointCloud2>("/kitti/velo/pointcloud", 10, &RosSub_Pub::callback, this);
        //sub_1=n.subscribe<sensor_msgs::PointCloud2>("/lidar_top",10, &RosSub_Pub::callback, this);
        pub_bbox = n.advertise<jsk_recognition_msgs::BoundingBoxArray>("/boxes", 10,true);
    }  
      
    void callback(const sensor_msgs::PointCloud2ConstPtr& msg)  
    {  
       
       clock_t start_time=clock(); 
        pcl::PointCloud<pcl::PointXYZ> cloud;
        clock_t start_time1=clock(); 
        pcl::fromROSMsg(*msg, cloud);
        clock_t end_time1=clock();
        //std::cout<< "point receive time is: "<<static_cast<double>(end_time1-start_time1)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;//输出运行时间
    
        

     
        std::vector<int> points_shape;
      std::vector<float> points_data;
      start_time1=clock();
      if (!preprocess(cloud, 4, point_cloud_range, &points_shape, &points_data)) {
          LOG(ERROR) << "Failed to preprocess!\n";
          return;
      }
        end_time1=clock();
        //std::cout<< "point preprocess time is: "<<static_cast<double>(end_time1-start_time1)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;//输出运行时间
    
        std::vector<float> box3d_lidar;
        std::vector<int64_t> label_preds;
        std::vector<float> scores;
        start_time1=clock();
        run(predictor.get(), points_shape, points_data, &box3d_lidar, &label_preds,&scores);
        end_time1=clock();
        //std::cout<< "running time is: "<<static_cast<double>(end_time1-start_time1)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;//输出运行时间
        start_time1=clock();
        jsk_recognition_msgs::BoundingBoxArray arr_bbox;
        int num_bbox3d = scores.size();
        for (size_t box_idx = 0; box_idx != num_bbox3d; ++box_idx) {
             // filter fake results:  score = -1
            if (scores[box_idx] < 0) {
            continue;
            }
            jsk_recognition_msgs::BoundingBox bbox;
            bbox.header.frame_id = msg->header.frame_id;  // Replace with your frame_id
            bbox.header.stamp = msg->header.stamp;
            bbox.pose.position.x =  box3d_lidar[box_idx * 7 + 0];
            bbox.pose.position.y =  box3d_lidar[box_idx * 7 + 1];
            bbox.pose.position.z = box3d_lidar[box_idx * 7 + 2];
            bbox.dimensions.x = box3d_lidar[box_idx * 7 + 3];  // width
            bbox.dimensions.y = box3d_lidar[box_idx * 7 + 4];  // length
            bbox.dimensions.z = box3d_lidar[box_idx * 7 + 5];  // height
            // Using tf::Quaternion for quaternion from roll, pitch, yaw
            tf::Quaternion q = tf::createQuaternionFromRPY(0, 0, box3d_lidar[box_idx * 7 + 6]);
            bbox.pose.orientation.x = q.x();
            bbox.pose.orientation.y = q.y();
            bbox.pose.orientation.z = q.z();
            bbox.pose.orientation.w = q.w();
            bbox.value = scores[box_idx];
            bbox.label = label_preds[box_idx];
            arr_bbox.boxes.push_back(bbox);
            
        }
        end_time1=clock();
        //std::cout<< "bbox time is: "<<static_cast<double>(end_time1-start_time1)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;//输出运行时间
        
        

        //std::cout<<"find bbox Num:"<<arr_bbox.boxes.size()<<std::endl;
        arr_bbox.header.frame_id = msg->header.frame_id;
        arr_bbox.header.stamp = msg->header.stamp;
        start_time1=clock();
        pub_1.publish(msg);
        pub_bbox.publish(arr_bbox);
        end_time1=clock();
        //std::cout<< "publish time is: "<<static_cast<double>(end_time1-start_time1)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;//输出运行时间
    
        clock_t end_time=clock();
        std::cout<< "All Running time is: "<<static_cast<double>(end_time-start_time)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;//输出运行时间
    }  
      
private:  
    ros::NodeHandle n;   
    // 创建发布对象
    ros::Publisher pub_1; 
    // 创建订阅对象
    ros::Subscriber sub_1; 
    ros::Publisher pub_bbox;
      
};

int main(int argc,  char **argv)
{

    
    
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
        exit(EXIT_FAILURE);
    }
    
    point_cloud_range.push_back(0);
    point_cloud_range.push_back(-40);
    point_cloud_range.push_back(-3);
    point_cloud_range.push_back(70.4);
    point_cloud_range.push_back(40);
    point_cloud_range.push_back(1);
    std::string dataFile = "/home/nvidia/way/OpenPCDet/data/kitti/training/velodyne/000006.bin";
    unsigned int length = 0;
    void *pc_data = NULL;
    std::vector<int> points_shape;
    std::vector<float> points_data;
    if (!preprocess_raw(dataFile, 4, point_cloud_range,
                    &points_shape, &points_data)) {
        LOG(ERROR) << "Failed to preprocess!\n";
        
    }
    std::vector<float> box3d_lidar;
    std::vector<int64_t> label_preds;
    std::vector<float> scores;
    int num_bbox3d = scores.size();
    for(int i=0;i<20;i++){
      run(predictor.get(), points_shape, points_data, &box3d_lidar, &label_preds,&scores);
    }
    std::cout<<"*************warm up finish************"<<std::endl;
    if (predictor == nullptr) {
        return 0;
    }
    setlocale(LC_ALL,"");
    ros::init(argc,argv,"pointcloud_detector");
    RosSub_Pub Sub_pub_obj;
    ros::spin();
    return 0;
}