# Accel-Video-Pipe
AV Pipe :-)

## Todo

- [x] Preprocessing

- [ ] DNN Deployment
    - [x] LibTorch
    - [x] OpenVINO
    - [x] TVM
    
    | Engine   | Time/ms, BS=2 |
    | -------- | ------------- |
    | LibTorch | 320           |
    | OpenVINO | 170           |
    | TVM      | 900           |
    
- [ ] Pipelined Processing
    - [ ] Multi-threading
    - [ ] IPC method
    
- [x] Post Processing

```mermaid
graph TD
A[PreProcessing: Normalize]-->B[CNN Model]-->C[Gaussian Mod]-->D[Max Pred]
```