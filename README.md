AV Pipe :-)

## AV-Pipe Roadmap

* [x] Packaging NN inference engine, into universal API

  * [x] ONNX RT
  * [x] OpenVINO
  * [x] LibTorch
  * [ ] Cloud API

  TODO: 

  * multi-input/output optimization
  * **Processors** need a time indicator to allow repeative consuming

* [ ] Build Stream Package Struct/Class

  * [x] torch::Tensor
  * [x] cv::Mat
* [ ] Mat <-> Tensor
  * Attention to Sync problem

* [ ] CV vision transformation

  * Stream-generator
    * [ ] Video file
    * [ ] Webcam

  * Pre-processing
    * [ ] Normalization
    * [ ] Layout Transfromation
    * [ ] Rotation & Crop
  * Post-processing
    * [ ] Rendering


* [ ] YAML representation of Pipeline

* [ ] Code/Proc auto-generating

  ----- By 4.15 -----

* [ ] Pipeline Optimization

  * [ ] CPU:
    * [ ] Trade-off between #Threads vs. #Cores
  * [ ] Heterogeneous Arch:
    * [ ] Thread allocation & scheduling

* [ ] Extension

  * [ ] iOS
  * [ ] Android