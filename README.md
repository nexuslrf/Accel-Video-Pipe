AV Pipe :-)

## The Structure of Accel-Video-Pipe

Namespace: `avp`

Base Classes (in `avpipe/base.hpp`):

* `StreamPacket`: to store temporal data to be processed;
  * `mat`: `cv::Mat` type, used for opencv-related operations/computations;
  * `tensor`: `aT::Tensor` type (maybe the most friendly C++ tensor type), used for DNN-related operations.

* `Stream`: a queue of `StreamPacket`. support synchronized, atomic queue operations.

* `PipeProcessor`: the actual computing module; 
  * `init`: for initialization
  * `process`: a universal procedure for each computing module:
    * take `StreamPacket` from `inStreams`, prepare the `StreamPacket` for `outStreams`.
  * `run`: a virtual function must be implemented by different modules.
  * `bindStream`: used to bind `Stream` to the `PipeProcessor`. 

## AV-Pipe Roadmap

**TODO:** 

* don't think about *async iterator* right now!

* [ ] Build Stream Packet Struct/Class

  * [x] torch::Tensor
  * [x] cv::Mat
  * [x] Mat <-> Tensor
  * Attention to Sync problem

* [ ] Processing Components

  Three major func:

  * initialization
  * Stream binding
  * Processing
    
    * **Note:** right now each call **only Process one** packet.
    
  * [ ] (WIP) Combine/operator+
    
  * [x] Packaging NN inference engine, into universal API
    
      * [x] ONNX RT
      * [x] OpenVINO
      * [x] LibTorch
    * [ ] Cloud API: consider *openSSH*
    
    TODO: 
    
      * [x] multi-input/output optimization
    * [x] **Processors** need a time indicator to allow repeative consuming
    
  * [ ] CV vision transformations
    
      * [x] Stream-generator
        * [x] Video file
        * [x] Webcam
      * [ ] Pre-processing
        * [x] Normalization
        * [x] Layout Transfromation
        * [ ] Rotation
        * [x] Crop
      * [ ] Post-processing
        * [ ] Rendering
          * [x] LandMarks
          * [ ] Bounding Boxes
      
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