## Todo

- [x] Preprocessing

- [x] DNN Deployment
    - [x] LibTorch
    - [x] OpenVINO
    - [x] TVM
    - [x] ONNX
    
    | Engine   | Time/ms, BS=2 |
    | -------- | ------------- |
    | LibTorch | 320           |
    | OpenVINO | 150           |
    | TVM      | 600           |
    | ONNX RT  | 300           |
    
- [ ] Pipelined Processing
    - [ ] Multi-threading
    - [ ] IPC method
    
- [x] Post Processing



Pose Pipe:

```mermaid
graph TD
A[PreProcessing: Normalize]-->B[CNN Model]-->C[Gaussian Mod]-->D[Max Pred]
```

Palm Detection

````mermaid
graph TD
A[PreProc: Normalize]-->B[NN Model]-->C{raw box}
B-->D{raw score}
C-->E[decode box]
D-->F[sigmoid]
E-->G[masking]
F-->G
G-->H[weighted_NMS]
````

Hand 

```mermaid
graph TD
A[Video]-->B[Palm Detection Network]
B-->C[Rotation and Cropping]
C-->D[Hand Landmark Network]
D-->E[Show Results]
D-->B
A-->D
```

TensorRT workflow

```mermaid
graph TD
A[initializeSampleParams, file_dir, bs, null_engine etc.]-->B[Build#1: init builder, network, config, onnxparser]
B-->C[Build#2: construct: Network, mEngine]
C-->D[Infer#1: BufferManager, context, processInput to Device]
D-->E[Infer#2: Execute, copyOutputToHost]

```

| | Has (recent)c++ prebuilts | C++ buildable | Handle PreBuilt Models | HW Acceleration |
| ------------------------- | ------------- | ---------------------- | --------------- | ---- |
| MXNET                     | NO            | Not So Far             | Yes | Fastest Cuda impl |
| TensorFlow | NO | Not since change to bazel | The pgm is the model | Slower |
| CNTK |  |  |  |  |
| caffe2 | Yes | Yes | Yes lots of them | yes |
| libtorch | Yes | Yes | Only TorchScript | Yes |
| OpenVino | Yes | Yes | Yes lots | Yes, Intel only |
