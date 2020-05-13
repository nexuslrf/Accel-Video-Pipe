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

* [x] Build Stream Packet Struct/Class

  * [x] torch::Tensor
  * [x] cv::Mat
  * [x] Mat <-> Tensor
  * Attention to Sync problem

* [x] Processing Components

  Three major func:

  * initialization
  * Stream binding
  * Processing
    
    * **Note:** right now each call **only Process one** packet.

  **TODO:**

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
    
  * [x] CV vision transformations
    
      * [x] Stream-generator
        * [x] Video file
        * [x] Webcam
      * [x] Pre-processing
        * [x] Normalization
        * [x] Layout Transfromation
        * [x] Rotation
        * [x] Crop
      * [x] Post-processing
        * [x] Rendering
          * [x] LandMarks
          * [x] Bounding Boxes
      
  
* [x] Samples:

  * [x] Pose Estimation
  * [x] Hand Tracking

------

* [x] Automation: use Python here...
  * [x] Visualization
  * [x] Code auto-gen
    * [x] Yaml to code:
      * [x] How to pass config into PipeProcessor:
        * [x] <s>add a another initial function in hpp?</s> ❌
        * [x] follow the default-yaml to pass all params to PipeProcessor? Make sure default-cfg has exactly the same order as the pipeProcessor's initial list. ✔️
    * [x] <s>Code to Yaml</s>
* [ ] Optimization
  * [ ] Multi-threading
    * [ ] make sure avp can run in multiple threads
      * [x] Pose_estimation
      * [ ] Hand
    * Problems:
      * Timing sync: Fast-Slow Problem: one thread consuming faster than another thread on the same stream, which causes time wasting? 

        Consider 3 solutions: 

        * [x] The faster one wait for the slower one ✔️
        * [x] <s>Async pointer for each pipe in each pipeprocessor: **TODO**</s>
          * [Behavior of deque iterator](https://stackoverflow.com/questions/10373985/c-deque-when-iterators-are-invalidated) makes it very hard to keep tracking... *Just give up*...
        * [x] Do not use one pipe as multiple inStreams across different threads.
          * Bi-directional or Uni-directional binding? Prefer **Uni-direction**..

      * [x] <s>Redundant Pipes: combining and wrapping</s>
        * How to init it? How to run it? What about the skipEmpty?
          * Only the first processor has instreams: Limited use cases...
        * Conclusion: Unimportant module with many issues to be pre-defined. **TODO** in future version
      * [x] Setting limits for stream capacity... Avoid unlimited packet feeding.
        * [x] If stream is full, make it sleep for a while...
      * [x] Need a thread-safe blocking queue, to reduce busy waiting. [ref](https://www.jianshu.com/p/c1dfa1d40f53)
        * [x] Add a getPacket method for Stream class: conditional variable!

      * [ ] A safe way to end the pipeline:  finish/over packet signal for packet
      * [ ] A better way to control how many frames processed?
      * [ ] Multi-Threading timing

    * Goal: Given timing info of each pipeProcessor and maximal number of useful phy threads, how to automatically schedule the pipe graph onto different threads?

      *Prototype Principles:*

      1. Find the critical path
      2. Find the most time consuming processors, try to pipeline them into different stages
      3. #Threads < Max#Threads
      4. Pay great attention to dependencies!
  * [ ] Heterogeneous Arch:
    
    * [ ] Thread allocation & scheduling

-----------

* [ ] Extension

  * [ ] iOS
  * [ ] Android