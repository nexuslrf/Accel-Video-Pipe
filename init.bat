"\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"
set "MSBUILD_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe"
set "Path=C:\Users\Ruofan\Programming\Libraries\libtorch\lib;C:\Users\Ruofan\Programming\Libraries\onnxruntime-win-x64-1.1.2\lib;C:\Users\Ruofan\Programming\Libraries\glog-0.4.0\Release;%Path%"
"%MSBUILD_BIN%" video_analysis.sln /p:Configuration=Release
