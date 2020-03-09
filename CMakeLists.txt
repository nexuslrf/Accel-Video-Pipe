cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(main)

if (WIN32)
 string(APPEND CMAKE_CXX_FLAGS " /W4")
else()
 string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
 string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
endif()

set(InferenceEngine_DIR "E:\\intelswtool\\openvino_2020.1.033\\deployment_tools\\inference_engine\\share")
set(CMAKE_PREFIX_PATH "E:\\workspace\\daka\\libtorch-win-shared-with-deps-1.4.0\\libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(OpenCV_DIR "E:\\intelswtool\\openvino_2020.1.033\\opencv\\cmake")
find_package(OpenCV REQUIRED)

add_executable(main video-analysis.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
target_link_libraries(main  "${OpenCV_LIBS}")
set_property(TARGET main PROPERTY CXX_STANDARD 14)

set(ONNXRUNTIME_ROOTDIR "E:\\workspace\\daka\\onnxruntime_RelWithDebInfo")
include_directories("${ONNXRUNTIME_ROOTDIR}/include" "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET main
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:main>)
endif (MSVC)
