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

* Redundant PackType

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

  **TODO:**

  * [ ] Combine/operator+: pending, *unimportant*
  * [x] Multiplexer: 
    * [x] Implemented by `TemplateProcessor`
  * [x] Placeholder to bypass empty checking
  * [x] 🌟Try to simplify the processing of user-defined processors:
    * [x] Lambda Function & Function Pointer
    * [x] Implemented by `TemplateProcessor`
  * [x] Redesign the empty check condition flow, to avoid partial empty cases, make sure all inStreams having the same behavior.    
  * [x] When auto-generate pipeline, make sure to scan all input Streams to avoid empty streams. `nullPacket` is a necessary placeholder!
  * [ ] Find a better way to output debug info/logs: consider [glog](https://github.com/google/glog)!

  Processors:

  * [x] Packaging NN inference engine, into universal API
    
      * [x] ONNX RT
      * [x] OpenVINO
      * [x] LibTorch
    * [x] TensorRT
    * [ ] Cloud API: consider *gRPC*
    
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
        * [x] Rotation
        * [x] Crop
      * [ ] Post-processing
        * [ ] Rendering
          * [x] LandMarks
          * [x] Bounding Boxes
      

* [ ] Samples:

  * [x] Pose Estimation
  * [x] Hand Tracking

------

* [ ] Automation: use Python here...
  * [x] Visualization
  * [ ] Code auto-gen
    * [ ] Yaml to code:
      * [ ] How to pass config into PipeProcessor:
        * [x] add a another initial function in hpp? ❌
        * [ ] follow the default-yaml to pass all params to PipeProcessor? Make sure default-cfg has exactly the same order as the pipeProcessor's initial list. ✔️
    * [ ] Code to Yaml
* [ ] Pipeline Optimization
* [ ] CPU:
    * [ ] Trade-off between #Threads vs. #Cores
  * [ ] Heterogeneous Arch:
    * [ ] Thread allocation & scheduling

-----------

* [ ] Extension

  * [ ] iOS
  * [ ] Android