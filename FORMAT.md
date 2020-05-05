# Pipe Define Format in YAML

Here we use a opensourced c++ yaml parser [lib](https://github.com/jbeder/yaml-cpp) (or maybe using python to generate a skeleton and flow graph, if it is technically difficult to build with c++.

**Why yaml?** Cuz I frequently use yaml to operate k8s while I am writing AVP. It is simple and easy to read.

The major part of the stream pipe processing is to define and configure each PipeProcessors. This would be very similar with caffe's prototxt file (maybe...)

Useful link: [Grammar for YAML](https://www.runoob.com/w3cnote/yaml-intro.html), [blog](https://www.cnblogs.com/sddai/p/9626392.html)

Prototype yaml file:

```yaml
pipeProcessor: ONNXRuntimeProcessor
label: handDetection # label will also passed to pp_name 
args:
  dims: [1, 3, 256, 256]
  data_layout: NCHW
  model_path: ./net.onnx
  num_output: 2
binding: # indicate all inStreams
- label: normalization
  out_idx: 0 # may need optimized, add alias for out_idx
```

**Problems:**

* Unefficiency
* How deal with Lambda Functions
* Python or C++ auto-gen?

Long long way to go for the real implementation...