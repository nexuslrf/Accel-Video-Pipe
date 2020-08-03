cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
project( video_analysis )

set(CMAKE_BUILD_TYPE "Release")

set(OpenCV_DIR "C:\\Program Files (x86)\\IntelSWTools\\openvino\\opencv\\cmake")
set(CMAKE_PREFIX_PATH "C:\\Users\\shoot\\Programming\\Libraries\\libtorch")
set(InferenceEngine_DIR "C:\\Program Files (x86)\\IntelSWTools\\openvino\\deployment_tools\\inference_engine\\share")

set_property (DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS _CRT_SECURE_NO_WARNINGS)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS -DNOMINMAX")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc") #no asynchronous structured exception handling
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251 /wd4275 /wd4267") #disable some warnings

find_package(InferenceEngine REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
#include_directories("C:\\Users\\shoot\\Programming\\Libraries\\libtorch\\include")
include_directories("C:\\Users\\shoot\\Programming\\Libraries\\onnxruntime-win-x64-1.1.2\\include")

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

message(STATUS "OpenVINO status")
message(STATUS "    version: ${InferenceEngine_VERSION}")
message(STATUS "    libraries: ${InferenceEngine_LIBRARIES}")

add_executable( video-analysis video-analysis.cpp )
target_link_libraries( video-analysis ${OpenCV_LIBS})
target_link_libraries( video-analysis ${TORCH_LIBRARIES})

#add_definitions(-D_DEBUG)
#add_definitions(-D_FOO_TEST)
add_definitions(-D_TIMING)

set(IE_MAIN_SAMPLES_DIR ${CMAKE_CURRENT_BINARY_DIR})
add_executable( speed-bench speed-bench.cpp )
target_link_libraries( speed-bench ${OpenCV_LIBS} )
target_link_libraries( speed-bench ${TORCH_LIBRARIES})
target_link_libraries( speed-bench ${InferenceEngine_LIBRARIES})
target_link_libraries( speed-bench "C:\\Users\\shoot\\Programming\\Libraries\\onnxruntime-win-x64-1.1.2\\lib\\onnxruntime.lib")
set_target_properties( speed-bench PROPERTIES COMPILE_PDB_NAME speed-bench)

add_executable( palm-detection palm-detection.cpp)
target_link_libraries( palm-detection ${OpenCV_LIBS} )
target_link_libraries( palm-detection ${TORCH_LIBRARIES})
target_link_libraries( palm-detection "C:\\Users\\shoot\\Programming\\Libraries\\onnxruntime-win-x64-1.1.2\\lib\\onnxruntime.lib")
set_target_properties( palm-detection PROPERTIES COMPILE_PDB_NAME speed-bench)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl -pthread -fPIC -ltvm_runtime")


