# Accel-Video-Pipe
AV Pipe :-)

## Todo

- [x] Preprocessing

- [ ] DNN Deployment
    - [x] LibTorch
    - [ ] OpenVINO
    - [ ] TVM
    
    | Engine   | Time/ms, BS=2 |
    | -------- | ------------- |
    | LibTorch | 320           |
    | OpenVINO | 170           |
    |          |               |
    
- [ ] Pipelined Processing
    - [ ] Multi-threading
    - [ ] IPC method
    
- [x] Post Processing

```mermaid
graph TD
A[PreProcessing: Normalize]-->B[CNN Model]-->C[Gaussian Mod]-->D[Max Pred]
```