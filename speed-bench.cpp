// set "MSBUILD_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe"
// "%MSBUILD_BIN%" Samples.sln /p:Configuration=Release
#include <fstream>
#include <sstream>
#include <cmath>
#include <functional> 
#include <chrono>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <inference_engine.hpp>
// #include <dlpack/dlpack.h>
// #include <tvm/runtime/module.h>
// #include <tvm/runtime/registry.h>
// #include <tvm/runtime/packed_func.h>
#include <vector>
#include <onnxruntime_cxx_api.h>
// #ifdef CV_CXX11
#include <mutex>
#include <thread>
#include <queue>
// #endif

using namespace std;

inline void updInput(vector<torch::jit::IValue>& inputs, torch::Tensor& inTensor)
{
    inputs.pop_back();
    inputs.push_back(inTensor);
}

template<typename T>
void fillBlobRandom(InferenceEngine::Blob::Ptr& inputBlob) {
    InferenceEngine::MemoryBlob::Ptr minput = InferenceEngine::as<InferenceEngine::MemoryBlob>(inputBlob);
    if (!minput) {
        THROW_IE_EXCEPTION << "We expect inputBlob to be inherited from MemoryBlob in fillBlobRandom, "
            << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer happens
    auto minputHolder = minput->wmap();

    auto inputBlobData = minputHolder.as<T *>();
    for (size_t i = 0; i < inputBlob->size(); i++) {
        inputBlobData[i] = (T) rand() / RAND_MAX * 10;
    }
}

bool libtorch = false, 
    openvino_cpu = false, 
    openvino_gpu = false, 
    onnxrt = true;

// int modelWidth=256, modelHeight=256;
// int batchSize = 1, numJoints=17;
int modelWidth=192, modelHeight=320;
int batchSize = 1, numJoints=17;
int numLoop = 10;
string model_name = //"C:\\Users\\shoot\\Programming\\Accel-Video-Pipe\\models\\palm_detection";
    "C:\\Users\\shoot\\Programming\\CV_Experiments\\yolov3\\weights\\yolov3-tiny";
int main()
{
    // auto tmp = torch::from_file("anchors.bin", NULL, 2944*4, torch::kFloat32).reshape({2944, 4});
    // cout<<tmp.slice(0,0,6)<<endl;

    // fstream fin("anchors.bin", ios::in | ios::binary);
    // auto anchors = torch::empty({2944, 4}, torch::kFloat32);
    // fin.read((char *)anchors.data_ptr(), anchors.numel() * sizeof(float));
    // cout<<anchors.slice(0,0,6)<<endl;
    cout<<"Start!\n";
    auto veriTensor = torch::randn({batchSize, 3, modelHeight, modelWidth}, torch::kFloat32);
    // /* LibTorch */
    if(libtorch)
    {
        torch::NoGradGuard no_grad; // Disable back-grad buffering
        torch::Tensor inTensor, rawOut;
        vector<torch::jit::IValue> inputs;
        inputs.push_back(inTensor);
        torch::jit::script::Module model;
        model = torch::jit::load(model_name+".zip");
        for(int i=0; i<numLoop; i++)
        {
            auto stop1 = chrono::high_resolution_clock::now(); 
            inTensor = torch::randn({batchSize, 3, modelHeight, modelWidth}, torch::kFloat32);
            updInput(inputs, inTensor);
            rawOut = model.forward(inputs).toTensor();
            auto stop2 = chrono::high_resolution_clock::now(); 
            cout<<"LibTorch Processing Time: "<<chrono::duration_cast<chrono::milliseconds>(stop2-stop1).count()<<"ms\n";
        }
        updInput(inputs, veriTensor);
        rawOut = model.forward(inputs).toTensor();
        cout<<"Veri SUM: "<<rawOut.sum()<<endl;
    }

    /* OpenVINO */
    // CPU Version
    if(openvino_cpu)
    {
        string model_xml = model_name+"_fp16.xml";
        string model_bin = model_name+"_fp16.bin";
        InferenceEngine::Core ie;
        auto network = ie.ReadNetwork(model_xml, model_bin);
        network.setBatchSize(batchSize);
        auto input_info = network.getInputsInfo().begin()->second;
        string input_name = network.getInputsInfo().begin()->first;
        auto output_info = network.getOutputsInfo().begin()->second;
        string output_name = network.getOutputsInfo().begin()->first;
        auto executable_network = ie.LoadNetwork(network, "CPU"); // Unknown problems for GPU version
        auto infer_request = executable_network.CreateInferRequest();
        auto tDesc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32,
                                        {(uint)batchSize, 3, (uint)modelHeight, (uint)modelWidth},
                                        InferenceEngine::Layout::NCHW);
        cout<<"Start Infer?\n";
        for(int i=0; i<numLoop; i++)
        {                                  
            auto stop1 = chrono::high_resolution_clock::now();                                
            auto inData = torch::randn({batchSize,3,modelHeight,modelWidth}, torch::kF32);
            InferenceEngine::Blob::Ptr inBlob = InferenceEngine::make_shared_blob<float_t>(tDesc, (float_t*)inData.data_ptr());
            infer_request.SetBlob(input_name, inBlob);
            infer_request.Infer();
            InferenceEngine::Blob::Ptr output = infer_request.GetBlob(output_name);
            auto stop2 = chrono::high_resolution_clock::now(); 
            cout<<"OpenVINO CPU Processing Time: "<<chrono::duration_cast<chrono::milliseconds>(stop2-stop1).count()<<"ms\n";
        }
        InferenceEngine::Blob::Ptr inBlob = InferenceEngine::make_shared_blob<float_t>(tDesc, (float_t*)veriTensor.data_ptr());
        infer_request.SetBlob(input_name, inBlob);
        infer_request.Infer();
        InferenceEngine::Blob::Ptr output = infer_request.GetBlob(output_name);
        auto outTensor = torch::from_blob(output->buffer().as<float*>(), {batchSize,2944,18});
        cout<<"Veri SUM: "<<outTensor.sum()<<endl;
    }

    /* OpenVINO */
    // GPU Version
    if(openvino_gpu)
    {
        string model_xml = model_name + "_fp16.xml";
        string model_bin = model_name + "_fp16.bin";
        InferenceEngine::Core ie;
        auto network = ie.ReadNetwork(model_xml, model_bin);
        network.setBatchSize(batchSize);
        auto input_info = network.getInputsInfo().begin()->second;
        string input_name = network.getInputsInfo().begin()->first;
        auto output_info = network.getOutputsInfo().begin()->second;
        string output_name = network.getOutputsInfo().begin()->first;
        auto executable_network = ie.LoadNetwork(network, "GPU"); // Unknown problems for GPU version
        auto infer_request = executable_network.CreateInferRequest();
        auto tDesc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32,
                                                 {(uint)batchSize, 3, (uint)modelHeight, (uint)modelWidth},
                                                 InferenceEngine::Layout::NCHW);
        for (int i = 0; i < numLoop; i++)
        {
            auto stop1 = chrono::high_resolution_clock::now();
            auto inData = torch::randn({batchSize, 3, modelHeight, modelWidth}, torch::kF32);
            InferenceEngine::Blob::Ptr inBlob = InferenceEngine::make_shared_blob<float_t>(tDesc, (float_t *)inData.data_ptr());
            infer_request.SetBlob(input_name, inBlob);
            infer_request.Infer();
            InferenceEngine::Blob::Ptr output = infer_request.GetBlob(output_name);
            auto stop2 = chrono::high_resolution_clock::now();
            cout << "OpenVINO GPU Processing Time: " << chrono::duration_cast<chrono::milliseconds>(stop2 - stop1).count() << "ms\n";
        }
        InferenceEngine::Blob::Ptr inBlob = InferenceEngine::make_shared_blob<float_t>(tDesc, (float_t*)veriTensor.data_ptr());
        infer_request.SetBlob(input_name, inBlob);
        infer_request.Infer();
        InferenceEngine::Blob::Ptr output = infer_request.GetBlob(output_name);
        auto outTensor = torch::from_blob(output->buffer().as<float*>(), {batchSize,2944,18});
        cout<<"Veri SUM: "<<outTensor.sum()<<endl;
    }

    /* OpenVINO */
    // VPU Version
    // {
    //     string model_xml = model_name + "_fp16.xml";
    //     string model_bin = model_name + "_fp16.bin";
    //     InferenceEngine::Core ie;
    //     auto network = ie.ReadNetwork(model_xml, model_bin);
    //     network.setBatchSize(batchSize);
    //     auto input_info = network.getInputsInfo().begin()->second;
    //     string input_name = network.getInputsInfo().begin()->first;
    //     auto output_info = network.getOutputsInfo().begin()->second;
    //     string output_name = network.getOutputsInfo().begin()->first;
    //     auto executable_network = ie.LoadNetwork(network, "MYRIAD"); // Unknown problems for GPU version
    //     auto infer_request = executable_network.CreateInferRequest();
    //     auto tDesc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32,
    //                                              {(uint)batchSize, 3, (uint)modelHeight, (uint)modelWidth},
    //                                              InferenceEngine::Layout::NCHW);
    //     for (int i = 0; i < numLoop; i++)
    //     {
    //         auto stop1 = chrono::high_resolution_clock::now();
    //         auto inData = torch::randn({batchSize, 3, modelHeight, modelWidth}, torch::kF32);
    //         InferenceEngine::Blob::Ptr inBlob = InferenceEngine::make_shared_blob<float_t>(tDesc, (float_t *)inData.data_ptr());
    //         infer_request.SetBlob(input_name, inBlob);
    //         infer_request.Infer();
    //         InferenceEngine::Blob::Ptr output = infer_request.GetBlob(output_name);
    //         auto stop2 = chrono::high_resolution_clock::now();
    //         cout << "OpenVINO VPU Processing Time: " << chrono::duration_cast<chrono::milliseconds>(stop2 - stop1).count() << "ms\n";
    //     }
    //     InferenceEngine::Blob::Ptr inBlob = InferenceEngine::make_shared_blob<float_t>(tDesc, (float_t*)veriTensor.data_ptr());
    //     infer_request.SetBlob(input_name, inBlob);
    //     infer_request.Infer();
    //     InferenceEngine::Blob::Ptr output = infer_request.GetBlob(output_name);
    //     auto outTensor = torch::from_blob(output->buffer().as<float*>(), {1,2944,18});
    //     cout<<"Veri SUM: "<<outTensor.sum()<<endl;
    // }

    /* TVM */
    // {
    //     string module_file = model_name+".so";
    //     string module_param = model_name+".params";
    //     string module_json = model_name+".json";
    //     tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile(module_file);
    //     ifstream json_in(module_json, ios::in);
    //     string json_data((istreambuf_iterator<char>(json_in)), istreambuf_iterator<char>());
    //     json_in.close();
    //     ifstream params_in(module_param, ios::binary);
    //     string params_data((istreambuf_iterator<char>(params_in)), istreambuf_iterator<char>());
    //     params_in.close();
    //     TVMByteArray params_arr;
    //     params_arr.data = params_data.c_str();
    //     params_arr.size = params_data.length();

    //     int dtype_code = kDLFloat;
    //     int dtype_bits = 32;
    //     int dtype_lanes = 1;
    //     int device_type = kDLMetal;
    //     int device_id = 0;

    //     tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))
    //         (json_data, mod_dylib, device_type, device_id);
        
    //     DLTensor *x, *y;
    //     int in_ndim = 4, out_ndim = 4;
    //     int64_t in_shape[4] = {batchSize, 3, 256, 192}, out_shape[4] = {batchSize, 17, 64, 48};
    //     cout<<"Init\n";
    //     TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, kDLCPU, device_id, &x);
    //     TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, kDLCPU, device_id, &y);

    //     auto set_input = mod.GetFunction("set_input");
    //     auto load_params = mod.GetFunction("load_params");
    //     auto run = mod.GetFunction("run");
    //     auto get_output = mod.GetFunction("get_output");
    //     load_params(params_arr);
    //     cout<<"start run\n";
    //     for(int i=0 ;i< numLoop; i++)
    //     {
    //         auto stop1 = chrono::high_resolution_clock::now(); 
    //         auto inData = torch::randn({batchSize,3,modelHeight,modelWidth}, torch::kF32);
    //         memcpy(x->data, inData.data_ptr(), inData.numel()*sizeof(float));
    //         set_input("input", x);
    //         run();
    //         get_output(0, y);
    //         auto stop2 = chrono::high_resolution_clock::now(); 
    //         cout<<"TVM Processing Time: "<<chrono::duration_cast<chrono::milliseconds>(stop2-stop1).count()<<"ms\n";
    //     }

    // }

    /* ONNX Runtime */
    // Ref: https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
    if(onnxrt)
    {
        // initialize  enviroment...one enviroment per process
        // enviroment maintains thread pools and other state info
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

        // initialize session options if needed
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        string tmp = model_name + ".onnx";
        wstring model_path(tmp.begin(), tmp.end());
        cout<<"Hello?#1 "<<tmp<<"\n";
        Ort::Session session(env, model_path.c_str(), session_options);
        cout<<"Session Complete!\n";
        //*************************************************************************
        // print model input layer (node names, types, shape etc.)
        Ort::AllocatorWithDefaultOptions allocator;

        // print number of model input nodes
        size_t num_input_nodes = session.GetInputCount();
        std::vector<int64_t> input_node_dims = {batchSize, 3, modelHeight, modelWidth};  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                                // Otherwise need vector<vector<>>

        size_t input_tensor_size = batchSize * 3 * modelHeight * modelWidth;
        std::vector<float> input_tensor_values(input_tensor_size);
        std::vector<const char*> input_node_names = {"images"};
        std::vector<const char*> output_node_names = {"classes", "boxes"};

        // initialize input data with values in [0.0, 1.0]
        

        // create input tensor object from data values
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // score model & input tensor, get back output tensor
        // assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
        cout<<"Start ONNXRT!\n";
        for(int i=0; i<numLoop; i++)
        {                                  
            auto stop1 = chrono::high_resolution_clock::now();                                
            auto inData = torch::randn({batchSize,3,modelHeight,modelWidth}, torch::kF32);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float_t*)inData.data_ptr(), input_tensor_size, input_node_dims.data(), 4);
            cout<<"Start infer?\n";
            auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
            auto stop2 = chrono::high_resolution_clock::now(); 
            cout<<"ONNX Processing Time: "<<chrono::duration_cast<chrono::milliseconds>(stop2-stop1).count()<<"ms\n";
        }
    }
}