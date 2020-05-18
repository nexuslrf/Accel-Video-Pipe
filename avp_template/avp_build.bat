@echo off

set src_dir=%1
set exec_name=%2

if "%src_dir%"=="" (
    echo "Please specify src_dir!"
    exit
)

cd %src_dir%
cd

:: init build folder
if exist "build" (
    echo Found the build folder.
) else (
    echo Make build folder.
    mkdir build
)
cd build
:: Start cmake
cmake ..
:: Setup MSVS Build
set "MSBUILD_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe"
:: set "Path=C:\Users\Ruofan\Programming\Libraries\libtorch\lib;C:\Users\Ruofan\Programming\Libraries\onnxruntime-win-x64-1.1.2\lib;C:\Users\Ruofan\Programming\Libraries\glog-0.4.0\Release;%Path%"
"%MSBUILD_BIN%" %exec_name%.sln /p:Configuration=Release