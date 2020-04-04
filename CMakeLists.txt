cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_PREFIX_PATH "/Users/liangruofan1/Program/CNN_Deployment/libtorch")
project( video-analysis )

set(CMAKE_BUILD_TYPE "Debug")

find_package( InferenceEngine REQUIRED)
find_package( OpenCV 4.2 REQUIRED )
find_package( Torch REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${})
include_directories( "/Users/liangruofan1/Program/tvm/include")
include_directories( "/Users/liangruofan1/Program/tvm/3rdparty/dlpack/include")
include_directories( "/Users/liangruofan1/Program/tvm/3rdparty/dmlc-core/include")
include_directories( "/Users/liangruofan1/Program/CNN_Deployment/onnxruntime-osx-x64-1.1.2/include")
link_directories( "/Users/liangruofan1/Program/tvm/build")

add_executable( video-analysis video-analysis.cpp )
target_link_libraries( video-analysis ${OpenCV_LIBS} )
target_link_libraries( video-analysis "${TORCH_LIBRARIES}")
#add_definitions(-D_DEBUG)
#add_definitions(-D_FOO_TEST)
add_definitions(-D_TIMING)

#set(IE_MAIN_SAMPLES_DIR ${CMAKE_CURRENT_BINARY_DIR})
add_executable( speed-bench speed-bench.cpp )
target_link_libraries( speed-bench ${OpenCV_LIBS} )
target_link_libraries( speed-bench ${TORCH_LIBRARIES})
target_link_libraries( speed-bench "${InferenceEngine_LIBRARIES}")
target_link_libraries( speed-bench "/Users/liangruofan1/Program/CNN_Deployment/onnxruntime-osx-x64-1.1.2/lib/libonnxruntime.1.1.2.dylib")

add_executable( palm-detection palm-detection.cpp)
target_link_libraries( palm-detection ${OpenCV_LIBS} )
target_link_libraries( palm-detection "${TORCH_LIBRARIES}")
target_link_libraries( palm-detection "/Users/liangruofan1/Program/CNN_Deployment/onnxruntime-osx-x64-1.1.2/lib/libonnxruntime.1.1.2.dylib")

add_executable( hand-keypoint hand-keypoint.cpp)
target_link_libraries( hand-keypoint ${OpenCV_LIBS} )
target_link_libraries( hand-keypoint "${TORCH_LIBRARIES}")
target_link_libraries( hand-keypoint "/Users/liangruofan1/Program/CNN_Deployment/onnxruntime-osx-x64-1.1.2/lib/libonnxruntime.1.1.2.dylib")

add_executable( multi-model multi-model.cpp)
target_link_libraries( multi-model ${OpenCV_LIBS} )
target_link_libraries( multi-model "${TORCH_LIBRARIES}")
target_link_libraries( multi-model "/Users/liangruofan1/Program/CNN_Deployment/onnxruntime-osx-x64-1.1.2/lib/libonnxruntime.1.1.2.dylib")


message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

message(STATUS "OpenVINO status")
message(STATUS "    version: ${InferenceEngine_VERSION}")
message(STATUS "    libraries: ${InferenceEngine_LIBRARIES}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl -pthread -fPIC -ltvm_runtime")
