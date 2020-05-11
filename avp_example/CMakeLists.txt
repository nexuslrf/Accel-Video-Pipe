cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_PREFIX_PATH "/Users/liangruofan1/Program/CNN_Deployment/libtorch")
project( avp-func-test )

set(CMAKE_BUILD_TYPE "Debug")

find_package( InferenceEngine REQUIRED)
find_package( OpenCV 4.2 REQUIRED )
find_package( Torch REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( "/Users/liangruofan1/Program/CNN_Deployment/onnxruntime-osx-x64-1.1.2/include")
include_directories( "../include")

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

message(STATUS "OpenVINO status")
message(STATUS "    version: ${InferenceEngine_VERSION}")
message(STATUS "    libraries: ${InferenceEngine_LIBRARIES}")

add_definition(-D_LOG_INFO)

add_executable( pose_estimation pose_estimation.cpp )
target_link_libraries( pose_estimation "${TORCH_LIBRARIES}")
target_link_libraries( pose_estimation ${OpenCV_LIBS} )
target_link_libraries( pose_estimation "${InferenceEngine_LIBRARIES}")
target_link_libraries( pose_estimation "/Users/liangruofan1/Program/CNN_Deployment/onnxruntime-osx-x64-1.1.2/lib/libonnxruntime.1.1.2.dylib")

add_executable( multi_hand_tracking multi_hand_tracking.cpp )
target_link_libraries( multi_hand_tracking "${TORCH_LIBRARIES}")
target_link_libraries( multi_hand_tracking ${OpenCV_LIBS} )
target_link_libraries( multi_hand_tracking "${InferenceEngine_LIBRARIES}")
target_link_libraries( multi_hand_tracking "/Users/liangruofan1/Program/CNN_Deployment/onnxruntime-osx-x64-1.1.2/lib/libonnxruntime.1.1.2.dylib")



#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl -pthread -fPIC")