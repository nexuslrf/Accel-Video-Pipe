@echo off
set src_dir=%1
set exec_name=%2

if "%src_dir%"=="" (
    echo "Please specify src_dir!"
    exit
)

:: init OpenVINO
call "\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"
set "Path=C:\Users\shoot\Programming\Libraries\libtorch\lib;C:\Users\shoot\Programming\Libraries\onnxruntime-win-x64-1.1.2\lib;C:\Users\shoot\Programming\Libraries\glog-0.4.0\Release;C:\Users\shoot\Programming\Libraries\ncnn\build-vs2019\install\lib;%Path%"

set PYTHONPATH

cd %src_dir%/build/Release
cd
%exec_name%.exe