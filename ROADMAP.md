# AV-Pipe Roadmap

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
  * [x] Find a better way to output debug info/logs: consider [glog](https://github.com/google/glog)!
  

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
    * [x] <s>Code to Yaml</s>: unnecessary
  
* [x] Optimization
  * [x] Multi-threading
    * [x] make sure avp can run in multiple threads
      * [x] Pose_estimation
      * [x] Hand
      * [ ] Face example
      * [ ] Yolo example
  
* Problems:
  
     * Timing sync: Fast-Slow Problem: one thread consuming faster than another thread on the same stream, which causes time wasting? 
     
       Consider 3 solutions: 
     
       * [x] The faster one wait for the slower one ✔️
       * [x] <s>Async pointer for each pipe in each pipeprocessor: **TODO **</s> ❌
         
         * [Behavior of deque iterator](https://stackoverflow.com/questions/10373985/c-deque-when-iterators-are-invalidated) makes it very hard to keep tracking... *Just give up*...
       * [x] Do not use one pipe as multiple inStreams across different threads.
          * Bi-directional or Uni-directional binding? Prefer **Uni-direction**.. ✔️
			* **Note:** for performance consideration, try this method.
           * When to add `coupledStreams`: the same outStream is consumed by multiple threads, including the thread generating this outStream.
       
     * [x] <s>Redundant Pipes: combining and wrapping</s> ❌
     
       * How to init it? How to run it? What about the skipEmpty?
         * Only the first processor has instreams: Limited use cases...
       * Conclusion: Unimportant module with many issues to be pre-defined. **TODO** in the future version
     
     * [x] Setting limits for stream capacity... Avoid unlimited packet feeding.
     
       * [x] If stream is full, make it sleep for a while...
     
     * [x] Need a thread-safe blocking queue, to reduce busy waiting. [ref1](https://www.jianshu.com/p/c1dfa1d40f53), [ref2](https://blog.csdn.net/big_yellow_duck/article/details/52601543)
     
     * [x] Add a getPacket method for Stream class: conditional variable!
     
     * [x] A safe way to end the pipeline:  finish/over packet signal for packet
     
     * [x] Multi-Threading timing:

       * Overall process time per frame: time the main thread. By blocking mechanism, the main thread timing is quite accurate.
         * Component processing time: include waiting time or not? Ans: **not include**
       * [ ] write timing info back to yaml file
     
       * [x] **Bug**: LibTorch's memory management, need to be refined!
     
       * `torch::from_blob` operator would overwrite the previous saved data_pointer
             * **solution:** when a tensor is used in a queue, use `torch::empty` and `memcpy ` to bypass.
     
     * [x] Try to generate a Gantt chart for demo? need to add **glog** support! 
     
       * [ ] **TODO**: need to generalize the scripts
     
       
     
* [x] Auto-Multi-threading:
  
    Goal: Given timing info of each pipeProcessor and maximal number of useful phy threads, how to automatically schedule the pipe graph onto different threads?
  
    *Prototype Principles:*  (Proven to be a NP-Complete problem. [ref1](https://en.wikipedia.org/wiki/Graph_partition), [ref2](http://www.cs.utexas.edu/~pingali/CS377P/2017sp/lectures/Algorithm%20abstractions.pdf))
  
    1. Find the most time consuming processors, try to pipeline them into different stages
  2. #Threads < Max#Threads
  3. Pay great attention to dependencies!
  
  *Heuristic Algorithm:* 
  
  1. If not specified, set the most time consuming PipeProcessor's timing as the thread_time_threshold.
  2. Put the whole orderings into the a single thread:
  3. while #threads < max#threads and #threads is growing:
     1. find the most time consuming thread in threads list, denoted as $thread_{max}$
     2. find the most time consuming pipeProcessor in $thread_{max}$, denoted as $proc_{max}$
     3. get $proc_{max}$'s $predecessors$, $successors$ and $unrelated\ nodes$.
     4. if #threads available >= 3:
        1. try to merge nodes in $predecessors$ and $successors$ in to $proc_{max}$ as long as the this set's consumed time not exceed thread_time_threshold.
        2. add 3 threads: $predecessors$, $successors$ and the set obtained from last step.
     5. if #threads available == 2:
        1. merge $proc_{max}$ with the less time consuming one between $predecessors$ and $successors$
        2. add 2 threads.

  * [x] Heterogeneous Arch:
    
    * [x] Thread allocation & scheduling

-----------

* [ ] Extension

  * [ ] iOS
  * [ ] Android