# Accel-Video-Pipe
AV Pipe :-)

## Todo

- [x] Preprocessing

- [ ] DNN Deployment
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