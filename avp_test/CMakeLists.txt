cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_PREFIX_PATH "C:\\Users\\Ruofan\\Programming\\Libraries\\libtorch")
project( avp-func-test )

set(CMAKE_BUILD_TYPE "Release")

set_property (DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS _CRT_SECURE_NO_WARNINGS)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS -DNOMINMAX")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc") #no asynchronous structured exception handling
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251 /wd4275 /wd4267") #disable some warnings

set(TENSORRT_ROOT "path\\to\\tensorrt")
set(ONNXRT_LIB "C:\\Users\\Ruofan\\Programming\\Libraries\\onnxruntime-win-x64-1.1.2\\lib\\onnxruntime.lib")

find_package( InferenceEngine REQUIRED)
find_package( OpenCV 4.2 REQUIRED )
find_package( Torch REQUIRED)

find_package( CUDA )
if(NOT CUDA_FOUND)
    message(STATUS "Your device doesn't support CUDA!")
else()
    add_definitions(-D_TENSORRT)
    set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3
        -gencode arch=compute_61,code=sm_61
    )
    # find TensorRT
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES include)
    MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
    find_library(TENSORRT_LIBRARY_INFER nvinfer
            HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 lib/x64)
    find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
            HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 lib/x64)
    set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN})
    MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
    find_package_handle_standard_args(
            TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
    if(NOT TENSORRT_FOUND)
        message(STATUS "Cannot find TensorRT library.")
    endif()
    LINK_LIBRARIES("/usr/lib/x86_64-linux-gnu/libnvonnxparser.so")
endif()

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories("C:\\Users\\Ruofan\\Programming\\Libraries\\onnxruntime-win-x64-1.1.2\\include")
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
target_link_libraries( nn_infer_test ${InferenceEngine_LIBRARIES})
target_link_libraries( nn_infer_test ${ONNXRT_LIB})
if(CUDA_FOUND)
    message(STATUS "Add Links for CUDA related libraries...")
    target_include_directories(nn_infer_test PUBLIC ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(nn_infer_test ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
endif()

# add_executable( cv_trans_test cv_trans_test.cpp )
# target_link_libraries( cv_trans_test "${TORCH_LIBRARIES}")
# target_link_libraries( cv_trans_test ${OpenCV_LIBS} )
# target_link_libraries( cv_trans_test ${InferenceEngine_LIBRARIES})
# target_link_libraries( cv_trans_test ${ONNXRT_LIB})

# add_executable( packet_test packet_test.cpp )
# target_link_libraries( packet_test "${TORCH_LIBRARIES}")
# target_link_libraries( packet_test "${OpenCV_LIBS}" )
# target_link_libraries( packet_test ${InferenceEngine_LIBRARIES})
# target_link_libraries( packet_test ${ONNXRT_LIB})

# add_executable( util_proc_test util_proc_test.cpp )
# target_link_libraries( util_proc_test "${TORCH_LIBRARIES}")
# target_link_libraries( util_proc_test "${OpenCV_LIBS}" )
# target_link_libraries( util_proc_test ${InferenceEngine_LIBRARIES})

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl -pthread -fPIC")