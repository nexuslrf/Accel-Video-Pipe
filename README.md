AV Pipe :-)

## AV-Pipe Roadmap

**TODO:** 

* don't think about *async iterator* right now!

* [ ] Build Stream Packet Struct/Class

  * [x] torch::Tensor
  * [x] cv::Mat
  * [ ] Mat <-> Tensor
  * Attention to Sync problem

* [ ] Processing Components

  Three major func:

  * initialization
  * Stream binding
  * Processing
    
    * **Note:** right now each call **only Process one** packet.
* (WIP) Combine/operator+
  
* [x] Packaging NN inference engine, into universal API
  
    * [x] ONNX RT
    * [x] OpenVINO
    * [x] LibTorch
  * [ ] Cloud API
  
  TODO: 
  
    * multi-input/output optimization
  * **Processors** need a time indicator to allow repeative consuming
  
* [ ] CV vision transformations
  
    * [x] Stream-generator
      * [x] Video file
      * [x] Webcam
    * [ ] Pre-processing
      * [x] Normalization
      * [ ] Layout Transfromation
      * [ ] Rotation
      * [x] Crop
    * [ ] Post-processing
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