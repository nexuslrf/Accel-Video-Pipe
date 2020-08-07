cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_PREFIX_PATH "C:\\Users\\shoot\\Programming\\Libraries\\libtorch")
project( avp-example )

set(CMAKE_BUILD_TYPE "Release")

set_property (DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS _CRT_SECURE_NO_WARNINGS)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS -DNOMINMAX")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc") #no asynchronous structured exception handling
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251 /wd4275 /wd4267") #disable some warnings

set(ONNXRT_LIB "C:\\Users\\shoot\\Programming\\Libraries\\onnxruntime-win-x64-1.1.2\\lib\\onnxruntime.lib")

find_package( InferenceEngine REQUIRED)
find_package( OpenCV 4.2 REQUIRED )
find_package( Torch 1.5 REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories("C:\\Users\\shoot\\Programming\\Libraries\\onnxruntime-win-x64-1.1.2\\include")
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

add_executable( pose_estimation pose_estimation.cpp )
target_link_libraries( pose_estimation "${TORCH_LIBRARIES}")
target_link_libraries( pose_estimation ${OpenCV_LIBS} )
target_link_libraries( pose_estimation "${InferenceEngine_LIBRARIES}")
target_link_libraries( pose_estimation ${ONNXRT_LIB})

add_executable( multi_hand_tracking multi_hand_tracking.cpp )
target_link_libraries( multi_hand_tracking "${TORCH_LIBRARIES}")
target_link_libraries( multi_hand_tracking ${OpenCV_LIBS} )
target_link_libraries( multi_hand_tracking "${InferenceEngine_LIBRARIES}")
target_link_libraries( multi_hand_tracking ${ONNXRT_LIB})

add_executable( yolov3 yolov3.cpp )
target_link_libraries( yolov3 "${TORCH_LIBRARIES}")
target_link_libraries( yolov3 ${OpenCV_LIBS} )
target_link_libraries( yolov3 "${InferenceEngine_LIBRARIES}")
target_link_libraries( yolov3 ${ONNXRT_LIB})

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl -pthread -fPIC")