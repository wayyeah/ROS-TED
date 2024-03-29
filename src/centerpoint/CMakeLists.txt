cmake_minimum_required(VERSION 3.0)
project(centerpoint CXX C)
set(CMAKE_CXX_STANDARD 14)
option(WITH_MKL        "Compile demo with MKL/OpenBlas support, default use MKL."       OFF)
option(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    ON)
option(USE_TENSORRT "Compile demo with TensorRT."   OFF)
set(CUSTOM_OPERATOR_FILES custom_ops/voxel/voxelize_op.cc;custom_ops/voxel/voxelize_op.cu;custom_ops/iou3d_nms/iou3d_cpu.cpp;custom_ops/iou3d_nms/iou3d_nms_api.cpp;custom_ops/iou3d_nms/iou3d_nms.cpp;custom_ops/iou3d_nms/iou3d_nms_kernel.cu;custom_ops/pointnet2/sampling_gpu.cu;custom_ops/pointnet2/sampling.cc;custom_ops/pointnet2/voxel_query_gpu.cu;custom_ops/pointnet2/voxel_query.cc;custom_ops/pointnet2/group_points.cc;custom_ops/pointnet2/group_points_gpu.cu)

execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpfullversion -dumpversion
                OUTPUT_VARIABLE GCC_VERSION)
string(REGEX MATCHALL "[0-9]+" GCC_VERSION_COMPONENTS ${GCC_VERSION})
list(GET GCC_VERSION_COMPONENTS 0 GCC_MAJOR)
list(GET GCC_VERSION_COMPONENTS 1 GCC_MINOR)
set(GCC_VERSION "${GCC_MAJOR}.${GCC_MINOR}")
if (GCC_VERSION LESS "8.0")
    set(CMAKE_CXX_FLAGS "-Wl,--no-as-needed")
endif()
find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  jsk_recognition_msgs
)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(external/boost)





if(WITH_GPU)
  find_package(CUDA REQUIRED)
  add_definitions("-DPADDLE_WITH_CUDA")
endif()

if(NOT WITH_STATIC_LIB)
  add_definitions("-DPADDLE_WITH_SHARED_LIB")
else()
  # PD_INFER_DECL is mainly used to set the dllimport/dllexport attribute in dynamic library mode.
  # Set it to empty in static library mode to avoid compilation issues.
  add_definitions("/DPD_INFER_DECL=")
endif()
set(PADDLE_LIB /home/nvidia/way/paddle_inference_install_dir)
set(DEMO_NAME main)
include_directories("/home/nvidia/way/paddle_inference_install_dir/paddle/include")
include_directories(
   ${catkin_INCLUDE_DIRS}
)
catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs pcl_ros
)

macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

if(NOT DEFINED PADDLE_LIB)
  message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
endif()
if(NOT DEFINED DEMO_NAME)
  message(FATAL_ERROR "please set DEMO_NAME with -DDEMO_NAME=demo_name")
endif()

include_directories("${PADDLE_LIB}/")
set(PADDLE_LIB_THIRD_PARTY_PATH "${PADDLE_LIB}/third_party/install/")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}onnxruntime/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}paddle2onnx/include")


link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/lib")
link_directories("${PADDLE_LIB}/paddle/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}onnxruntime/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}paddle2onnx/lib")

if (WIN32)
  add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
  option(MSVC_STATIC_CRT "use static C Runtime library by default" ON)
  if (MSVC_STATIC_CRT)
    if (WITH_MKL)
      set(FLAG_OPENMP "/openmp")
    endif()
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
    safe_set_static_flag()
    if (WITH_STATIC_LIB)
      add_definitions(-DSTATIC_LIB)
    endif()
  endif()
else()
  if(WITH_MKL)
    set(FLAG_OPENMP "-fopenmp")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 ${FLAG_OPENMP}")
endif()

if(WITH_GPU)
  if(NOT WIN32)
    set(CUDA_LIB "/usr/local/cuda/lib64/" CACHE STRING "CUDA Library")
  else()
    if(CUDA_LIB STREQUAL "")
      set(CUDA_LIB "C:\\Program\ Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\lib\\x64")
    endif()
  endif(NOT WIN32)
endif()

if (USE_TENSORRT AND WITH_GPU)
  set(TENSORRT_ROOT "" CACHE STRING "The root directory of TensorRT library")
  if("${TENSORRT_ROOT}" STREQUAL "")
      message(FATAL_ERROR "The TENSORRT_ROOT is empty, you must assign it a value with CMake command. Such as: -DTENSORRT_ROOT=TENSORRT_ROOT_PATH ")
  endif()
  set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
  set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib)
endif()

if (NOT WIN32)
  if (USE_TENSORRT AND WITH_GPU)
      include_directories("${TENSORRT_INCLUDE_DIR}")
      link_directories("${TENSORRT_LIB_DIR}")
  endif()
endif(NOT WIN32)

if(WITH_MKL)
  set(MATH_LIB_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}mklml")
  include_directories("${MATH_LIB_PATH}/include")
  if(WIN32)
    set(MATH_LIB ${MATH_LIB_PATH}/lib/mklml${CMAKE_STATIC_LIBRARY_SUFFIX}
                 ${MATH_LIB_PATH}/lib/libiomp5md${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    set(MATH_LIB ${MATH_LIB_PATH}/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
                 ${MATH_LIB_PATH}/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif()
  set(MKLDNN_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}mkldnn")
  if(EXISTS ${MKLDNN_PATH})
    include_directories("${MKLDNN_PATH}/include")
    if(WIN32)
      set(MKLDNN_LIB ${MKLDNN_PATH}/lib/mkldnn.lib)
    else(WIN32)
      set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libmkldnn.so.0)
    endif(WIN32)
  endif()
else()
  set(OPENBLAS_LIB_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}openblas")
  include_directories("${OPENBLAS_LIB_PATH}/include/openblas")
  if(WIN32)
    set(MATH_LIB ${OPENBLAS_LIB_PATH}/lib/openblas${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    set(MATH_LIB ${OPENBLAS_LIB_PATH}/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
  endif()
endif()

if(WITH_STATIC_LIB)
  set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  if(WIN32)
    set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif()
endif()


if (NOT WIN32)
  if (GCC_VERSION LESS "8.0")
      set(EXTERNAL_LIB ${EXTERNAL_LIB} "-lssl -lcrypto -lz -lleveldb -lsnappy")
  endif()
  set(EXTERNAL_LIB ${EXTERNAL_LIB} "-lrt -ldl -lpthread")
  set(DEPS ${DEPS}
      ${MATH_LIB} ${MKLDNN_LIB}
      glog gflags protobuf  xxhash
      ${EXTERNAL_LIB})
else()
  set(DEPS ${DEPS}
      ${MATH_LIB} ${MKLDNN_LIB}
      glog gflags_static libprotobuf  xxhash ${EXTERNAL_LIB})
  set(DEPS ${DEPS} shlwapi.lib)
endif(NOT WIN32)

if(WITH_GPU)
  if(NOT WIN32)
    if (USE_TENSORRT)
      set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/libnvinfer${CMAKE_SHARED_LIBRARY_SUFFIX})
      set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/libnvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
    set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
  else()
    if(USE_TENSORRT)
      set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/nvinfer${CMAKE_STATIC_LIBRARY_SUFFIX})
      set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/nvinfer_plugin${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif()
    set(DEPS ${DEPS} ${CUDA_LIB}/cudart${CMAKE_STATIC_LIBRARY_SUFFIX} )
    set(DEPS ${DEPS} ${CUDA_LIB}/cublas${CMAKE_STATIC_LIBRARY_SUFFIX} )
    set(DEPS ${DEPS} ${CUDA_LIB}/cudnn${CMAKE_STATIC_LIBRARY_SUFFIX} )
  endif()
endif()

cuda_add_library(pd_infer_custom_op ${CUSTOM_OPERATOR_FILES} SHARED)
add_executable(${DEMO_NAME} ${DEMO_NAME}.cpp)

if (GCC_VERSION GREATER_EQUAL "8.0")
    set(DEPS ${DEPS} libssl.a libcrypto.a libz.a libleveldb.a libsnappy.a)
endif()
set(DEPS ${DEPS} boost pd_infer_custom_op)# libssl.a libcrypto.a libz.a libleveldb.a libsnappy.a)

if(WIN32)
  if(USE_TENSORRT)
    add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${TENSORRT_LIB_DIR}/nvinfer${CMAKE_SHARED_LIBRARY_SUFFIX}
              ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
            COMMAND ${CMAKE_COMMAND} -E copy ${TENSORRT_LIB_DIR}/nvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX}
              ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
    )
  endif()
  if(WITH_MKL)
    add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy ${MATH_LIB_PATH}/lib/mklml.dll ${CMAKE_BINARY_DIR}/Release
          COMMAND ${CMAKE_COMMAND} -E copy ${MATH_LIB_PATH}/lib/libiomp5md.dll ${CMAKE_BINARY_DIR}/Release
          COMMAND ${CMAKE_COMMAND} -E copy ${MKLDNN_PATH}/lib/mkldnn.dll  ${CMAKE_BINARY_DIR}/Release
    )
  else()
    add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy ${OPENBLAS_LIB_PATH}/lib/openblas.dll ${CMAKE_BINARY_DIR}/Release
    )
  endif()
  if(NOT WITH_STATIC_LIB)
      add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_LIB}/paddle/lib/paddle_fluid.dll" ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
      )
  endif()
endif()


target_link_libraries(${DEMO_NAME} ${DEPS})
target_link_libraries(${DEMO_NAME} ${catkin_LIBRARIES})
