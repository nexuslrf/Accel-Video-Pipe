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

add_executable( nn_infer_test nn_infer_test.cpp )
target_link_libraries( nn_infer_test "${TORCH_LIBRARIES}")
target_link_libraries( nn_infer_test ${OpenCV_LIBS} )
target_link_libraries( nn_infer_test "${InferenceEngine_LIBRARIES}")
target_link_libraries( nn_infer_test "/Users/liangruofan1/Program/CNN_Deployment/onnxruntime-osx-x64-1.1.2/lib/libonnxruntime.1.1.2.dylib")

add_executable( cv_trans_test cv_trans_test.cpp )
target_link_libraries( cv_trans_test "${TORCH_LIBRARIES}")
target_link_libraries( cv_trans_test ${OpenCV_LIBS} )
target_link_libraries( cv_trans_test "${InferenceEngine_LIBRARIES}")
target_link_libraries( cv_trans_test "/Users/liangruofan1/Program/CNN_Deployment/onnxruntime-osx-x64-1.1.2/lib/libonnxruntime.1.1.2.dylib")

add_executable( packet_test packet_test.cpp )
target_link_libraries( packet_test "${TORCH_LIBRARIES}")
target_link_libraries( packet_test "${OpenCV_LIBS}" )
target_link_libraries( packet_test "${InferenceEngine_LIBRARIES}")
target_link_libraries( packet_test "/Users/liangruofan1/Program/CNN_Deployment/onnxruntime-osx-x64-1.1.2/lib/libonnxruntime.1.1.2.dylib")

add_executable( template_proc_test template_proc_test.cpp )
target_link_libraries( template_proc_test "${TORCH_LIBRARIES}")
target_link_libraries( template_proc_test "${OpenCV_LIBS}" )
target_link_libraries( template_proc_test "${InferenceEngine_LIBRARIES}")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl -pthread -fPIC")