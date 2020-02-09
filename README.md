# Accel-Video-Pipe
AV Pipe :-)

## Todo

- [ ] Preprocessing
- [ ] DNN Deployment
    - [ ] LibTorch
    - [ ] OpenVINO
    - [ ] TVM
- [ ] Pipelined Processing
    - [ ] Multi-threading
    - [ ] IPC method
- [ ] Post Processing

```mermaid
graph TD
A[PreProcessing: Normalize]-->B[CNN Model]-->C[Gaussian Mod]-->D[Max Pred]
```