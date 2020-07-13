## Intel® Edge AI for IoT Developers Nanodegree program

 
 <img src="Intel-Scholarship+2020@2x.jpg" width="500"/>         | <img src="Udacity.png" width="450"/>
    </br>
 

<div>
    <h1>Intel® Edge AI for IoT Developers Nanodegree Program</h1>
    <p>N.B.: Please don't use the assignment and quiz solution. Try to solve the problem by yourself.</p><br/>
    <p>Leverage the Intel® Distribution of OpenVINO™ Toolkit to fast-track development of high-performance computer vision and deep learning inference applications, and run pre-trained deep learning models for computer vision on-premise. You will identify key hardware specifications of various hardware types (CPU, VPU, FPGA, and Integrated GPU), and utilize the Intel® DevCloud for the Edge to test model performance on the various hardware types. Finally, you will use software tools to optimize deep learning models to improve performance of Edge AI systems. - <a href="https://www.udacity.com/course/intel-edge-ai-for-iot-developers-nanodegree--nd131">Source</a></p>
</div>

## Projects
## 1. [Deploy a People Counter App at the Edge](https://github.com/marwan1023/Intel-Edge-AI-for-IoT-Developers-Nanodegree-program/tree/master/Deploy%20a%20People%20Counter%20App%20at%20the%20Edge)

The project aims to create a people counting smart camera able to detect people using an optimized AI model at the Edge and extract relevant statistics like:

- Number of people on the captured video stream
- The duration they spend on screen
- Total people counted

These statistics are sent using JSON and MQTT to a server, for bandwidth saving enabling the use of the low-speed link. If needed is always possible to watch remotely the video stream for seeing what's is currently happening.

The challenges in this project are: select the right pre-trained model for doing the object detection, optimize the model to allow the inference on low-performance devices, properly adjust the input video stream using OpenCV for maximizing the model accuracy.

| Details            |              |
|-----------------------|---------------|
| Programming Language: |  Python 3.5 or 3.6 |


<img src="/Deploy a People Counter App at the Edge/images/people-counter-image.png" />

## 2. [Smart Queuing System](https://github.com/marwan1023/Intel-Edge-AI-for-IoT-Developers-Nanodegree-program/tree/master/Smart%20Queue%20Monitoring%20System)

The goal of this project is building an application to reduce congestion and queuing systems.
## The Scenarios
1. [Manufacturing Sector](https://github.com/marwan1023/marwan1023-Intel-Edge-AI-for-IoT-Developers-Nanodegree-program--to-confirm./blob/master/Smart%20Queue%20Monitoring%20System/Scenarios/Scenario%201.pdf)
2. [Retail Sector](https://github.com/marwan1023/marwan1023-Intel-Edge-AI-for-IoT-Developers-Nanodegree-program--to-confirm./blob/master/Smart%20Queue%20Monitoring%20System/Scenarios/Scenario%202.pdf)
3. [Transportation Sector](https://github.com/marwan1023/marwan1023-Intel-Edge-AI-for-IoT-Developers-Nanodegree-program--to-confirm./blob/master/Smart%20Queue%20Monitoring%20System/Scenarios/Scenario%203.pdf)
##  Instructions
- Propose a possible hardware solution for each scenario
- Build out your application and test its performance on the DevCloud using multiple hardware types
- Compare the performance to see which hardware performed best
- Revise your proposal based on the test results
## Requirements
### Hardware
- [DevCloud](https://devcloud.intel.com/edge/get_started/devcloud/)
### Software
- [Intel® Distribution of OpenVINO™ toolkit](https://docs.openvinotoolkit.org/2020.3/index.html)
### Model
- Download the person detection model from the Open Model Zoo 
  ```
  sudo /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name person-detection-retail-0013
  ```
## Results
1. Manufacturing Sector

   |  CPU   |  FPGA |  GPU  |  VPU  |
   |  :---: | :---: | :---: | :---: |    
   | <img src="/Smart Queue Monitoring System/results_gif/manufacturing/cpu/ezgif.com-video-to-gif_cpu.gif"/> | <img src="/Smart Queue Monitoring System/results_gif/manufacturing/fpga/ezgif.com-video-to-gif_Fpga.gif"/> | <img src="/Smart Queue Monitoring System/results_gif/manufacturing/gpu/ezgif.com-video-to-gif_gpu.gif"/> | <img src="/Smart Queue Monitoring System/results_gif/manufacturing/vpu/ezgif.com-video-to-gif_vpu.gif" /> |

2. Retail Sector

   |  CPU   |  FPGA |  GPU  |  VPU  |
   |  :---: | :---: | :---: | :---: |     
   | <img src="/Smart Queue Monitoring System/results_gif/retail/cpu/ezgif.com-video-to-gif_cpu.gif" />| <img src="/Smart Queue Monitoring System/results_gif/retail/fpga/ezgif.com-video-to-gif_fpga.gif" /> | <img src="/Smart Queue Monitoring System/results_gif/retail/gpu/ezgif.com-video-to-gif_gpu.gif"/> | <img src="/Smart Queue Monitoring System/results_gif/retail/vpu/ezgif.com-video-to-gif_vpu.gif"/> |

3. Transportation Sector

   |  CPU   |  FPGA |  GPU  |  VPU  |
   |  :---: | :---: | :---: | :---: |      
   | <img src="/Smart Queue Monitoring System/results_gif/transportation/cpu/ezgif.com-video-to-gif_cpu.gif" /> | <img src="/Smart Queue Monitoring System/results_gif/transportation/fpga/ezgif.com-video-to-gif_fpga.gif" /> | <img src="/Smart Queue Monitoring System/results_gif/transportation/gpu/ezgif.com-video-to-gif_gpu.gif" /> | <img src="/Smart Queue Monitoring System/results_gif/transportation/vpu/ezgif.com-video-to-gif_vpu.gif" /> |
   
## 3. [Computer Pointer Controller](https://github.com/marwan1023/Intel-Edge-AI-for-IoT-Developers-Nanodegree-program/tree/master/Computer%20Pointer%20Controller/starter)

In this project, you will use a Gaze Detection Model [Gaze Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html) to control the mouse pointer of your computer. 
You will be using the Gaze Estimation model to estimate the gaze of the user's eyes and change the mouse pointer position accordingly.
This project will demonstrate your ability to run multiple models in the same machine and coordinate the flow of data between those models.  
The gaze estimation model requires three inputs  you will have to use three other OpenVino models:

The head pose
The left eye image
The right eye image.

## Project requires and Installation
- Install intel distribution of openvino for Windows 10 [here](https://docs.openvinotoolkit.org/2020.2/_docs_install_guides_installing_openvino_windows.html)

To get these inputs, you will have to use three other OpenVino models:

* [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_0001_description_face_detection_adas_0001.html)
* [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
* [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)

### The Pipeline:
You will have to coordinate the flow of data from the input, and then amongst the different models and finally to the mouse controller. The flow of data will look like this:

![pipeline] <img src="/Computer Pointer Controller/starter/Images/pipeline.png" />

## Benchmarks

* I ran the model inference on CPU and GPU device on local machine given same input video and same virtual environment. Listed below are hardware versions:
    Model precisions tested:

  FP32
  FP16
  INT8
  Hardwares tested:

   CPU (2.3 GHz Intel Core i5)
   GPU (Intel(R) UHD Graphics 630)

I have checked Inference Time, Model Loading Time, and Frames Per Second model for FP16, FP32, and FP32-INT8


**Benchmark results of the model. CPU(FP32-INT8,FP16,FP32) and Asynchronous Inference**



<img src="/Computer Pointer Controller/starter/Images/cpu/model.cpu.png" />)  |  <img src="/Computer Pointer Controller/starter/Images/cpu/asycncpu.png" />
 :---------------------------------------------------------------------------:|:---------------------------------------:
<img src="/Computer Pointer Controller/starter/Images/cpu/asycncpu1.png" />   | <img src="/Computer Pointer Controller/starter/Images/cpu/asycncpu2.png" />)
</br>


**Benchmark results of the model. GPU(FP32-INT8,FP16,FP32) and Asynchronous Inference**


<img src="/Computer Pointer Controller/starter/Images/gpu/model.GPU.png" />  |  <img src="/Computer Pointer Controller/starter/Images/gpu/asyngpy.png" />
 :--------------------------------------------------------------------------:|:---------------------------------------:
<img src="/Computer Pointer Controller/starter/Images/gpu/asyngpy1.png" />   |  <img src="/Computer Pointer Controller/starter/Images/gpu/asyngpy2.png" />
</br>



* Due to non availability of FPGA and VPU in local machine, I did not run inference for these device types.



* FP32

  | Type of Hardware | Total inference time              | Total load time              | fps        |
  |------------------|-----------------------------------|------------------------------|------------|
  | CPU              |  31.6s                            | 0.930308s                    | 1.867089   |
  | GPU              |  32.8s                            | 33.834617s                   | 1.798780   |


* FP16
  
  
  
  | Type of Hardware | Total inference time              | Total load time               | fps       |
  |------------------|-----------------------------------|-------------------------------|-----------|
  | CPU              |  31.8s                            |  1.165073s                    | 1.855346  |
  | GPU              |  32.6s                            |  34.921903s                   | 1.809816  |




* FP32-INT8

  
  | Type of Hardware | Total inference time              | Total load time               | fps      |
  |------------------|-----------------------------------|-------------------------------|----------|
  | CPU              |  32.0s                            | 2.662999s                     | 1.843750 |
  | GPU              |  34.1s                            | 47.700375s                    | 1.730205 |


## Requirements
* 64-bit operating system that has 6th or newer generation of Intel processor running either Windows 10, Ubuntu 18.04.3 LTS, or macOS 10.13 or higher.
* [Installing OpenVINO (version 2020.1)](https://docs.openvinotoolkit.org/2020.1/index.html)
* [Installing Intel's Deep Learning Workbench (version 2020.1)](https://docs.openvinotoolkit.org/2020.1/_docs_Workbench_DG_Install_Workbench.html)
* [Installing Intel's VTune Amplifier](https://software.intel.com/en-us/get-started-with-vtune)


## Core Curriculum

### 1. Welcome to the Program

#### Lesson-1: Nanodegree Program Introduction
| No |                                                        Lesson                                                         |                                                                                                                                  Notes                                                                                                                                   |                         Link/Source                          |
|:--:|:---------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------:|
| 1 |                                                    Welcome to Udacity                                                  |                                                                                                                      Welcome note, Technology evolution                                                                                                                  |                       ------/------                          |
| 2 |                                        Welcome to the Nanodegree Program Experience                                    |                                                                                                                     Udacity mentor supprot, Helping tools                                                                                                                |                       ------/------                          |
| 3 |                                                    How to Succeed                                                      |                                                                                                                 Introduction of instructor, Goals (short or long term), Accountability, Learning strategies, Technical advice                                            |                       ------/------                          |
| 4 |                                      Welcome to Intel® Edge AI for IoT Developers                                      |                                                                                                                  Design, test, and deploy an edge AI application                                                                                                         |                       ------/------                          |
| 5 |                                           Prerequisites & Other Requirements                                           |                                                                                                                Python, Training and deploying deep learning models, Draining and deploying deep learning models, CLI, OpenCV                                             |                       ------/------                          |
| 6 |                                               Notebooks and Workspaces                                                 |                                                                                                                 Jupyter Notebooks, Jupyter Graffiti                                                                                                                      |                       ------/------                          |
| 7 |                                                  Graffiti Tutorial                                                     |                                                                                                                      Graffiti Tutorial                                                                                                                                   |                       ------/------                          |

### 2. Edge AI Fundamentals with OpenVINO

#### Lesson-1: Introduction to AI at the Edge
| No |                                                        Lesson                                                         |                                                                                                                                  Notes                                                                                                                                   |                         Link/Source                          |
|:--:|:---------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------:|
| 1 |                                                    Instructor Intro                                                    |                                                                                                                      Instructor pathway & introduction                                                                                                                   |                       ------/------                          |
| 2 |                                                 What is AI at the Edge?                                                |                                                                                                                 Edge means local (or near local) processing, Less impact on a network                                                                                    |                       ------/------                          |
| 3 |                                           Why is AI at the Edge Important?                                             |                                                                                                                 Network communication, Real-time processing, Sensitive data, Optimization software                                                                       |                       ------/------                          |
| 4 |                                           Applications of AI at the Edge                                               |                                                                                                                 Endless possibilities, IoT devices, Self driving, Animal tracking                                                                                        |                       ------/------                          |
| 5 |                                                   Historical Context                                                   |                                                                                                                    Historical background edge application                                                                                                                |                            [1]                               |
| 6 |                                                    Course Structure                                                    |                                                                                                                 Pre-trained models, Model optimizer, Inference engine, Deploying at the edge(handling input streams, processing model outputs, MQTT)                     |                            [2]                               |
| 7 |                                          Why Are the Topics Distinct?                                                  |                                                                                                                Train a model->Model optimizer->IR format->Inference engine->Edge application                                                                             |                       ------/------                          |
| 8 |                                             Relevant Tools and Prerequisites                                           |                                                                                                                Basics of computer vision and how AI models, Python or C++, Hardware & software requiremnts                                                               |                          [3 - 5]                             |
| 9 |                                                 What You Will Build                                                    |                                                                                                                 Build and deploy a People Counter App at the Edge                                                                                                        |                       ------/------                          |
|10 |                                                       Recap                                                            |                                                                                                                Basics of the edge, Importance of the edge and its history, Edge application                                                                              |                       ------/------                          |


#### Lesson-2: Leveraging Pre-Trained Models
| No |                                                        Lesson                                                         |                                                                                                                                  Notes                                                                                                                                   |                         Link/Source                          |
|:--:|:---------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------:|
| 1 |                                                      Introduction                                                      |                                                                                                                     Lesson objective                                                                                                                                     |                       ------/------                          |
| 2 |                                                 The OpenVINO™ Toolkit                                                  |                                                                                                                An open source library useful for edge deployment due to its performance maximizations and pre-trained models                                             |                       ------/------                          |
| 3 |                                            Pre-Trained Models in OpenVINO™                                             |                                                                                                                Model Zoo, in which the Free Model Set contains pre-trained models already converted using the Model Optimize                                             |                            [6]                               |
| 4 |                                           Types of Computer Vision Models                                              |                                                                                                                Classification, Detection, and Segmentation etc.                                                                                                          |                            [7]                               |
| 5 |                                           Case Studies in Computer Vision                                              |                                                                                                                     SSD, ResNet and MobileNet                                                                                                                            |                       ------/------                          |
| 6 |                                        Available Pre-Trained Models in OpenVINO™                                       |                                                                                                                 Public Model Set, Free Model Set                                                                                                                         |                            [6]                               |
| 7 |                                          Exercise: Loading Pre-Trained Models                                          |                                                                                                                 Find the Right Models, Download the Models, Verify the Downloads                                                                                         |                       ------/------                          |
| 8 |                                         Solution: Loading Pre-Trained Models                                           |                                                                                                                 Choosing Models, Downloading Models, Verifying Downloads                                                                                                 |                       ------/------                          |
| 9 |                                       Optimizations on the Pre-Trained Models                                          |                                                                                                                Dealt with different precisions of the different models                                                                                                   |                       ------/------                          |
|10 |                                        Choosing the Right Model for Your App                                           |                                                                                                                Try out different models for the application and a single use case                                                                                        |                       ------/------                          |
|11 |                                                Pre-processing Inputs                                                   |                                                                                                                Check out in any related documentation, Check color chanel, Input and output parameters                                                                   |                       ------/------                          |
|12 |                                          Exercise: Pre-processing Inputs                                               |                                                                                                                 Build preprocess_input file for processing the inputs parameters                                                                                         |                       ------/------                          |
|13 |                                          Solution: Pre-processing Inputs                                               |                                                                                                                   Solution of pre-processing inputs                                                                                                                      |                       ------/------                          |
|14 |                                             Handling Network Outputs                                                   |                                                                                                                Try out different models for the application and a single use case                                                                                        |                          [8 - 9]                             |
|15 |                                           Running Your First Edge App                                                  |                                                                                                                Load a pre-trained model into the Inference Engine, as well as call for functions to preprocess and handle the output in the appropriate locations        |                       ------/------                          |
|16 |                                       Exercise: Deploy An App at the Edge                                              |                                                                                                                 Implement the handling of the outputs of our three models                                                                                                |                       ------/------                          |
|17 |                                       Solution: Deploy An App at the Edge                                              |                                                                                                                 Car Meta Model, Pose Estimation, Text Detection Model Output Handling                                                                                    |                       ------/------                          |
|18 |                                                       Recap                                                            |                                                                                                                Basics of the Intel® Distribution of OpenVINO™ Toolkit, Different CV model types, Available Pre-Trained Model, Choosing the right Pre-Trained Model       |                       ------/------                          |
|19 |                                                Lesson Glossary                                                         |                                                                                                                Basics of the edge, Importance of the edge and its history, Edge application                                                                              |                       ------/------                          |

#### Lesson-3: The Model Optimizer
| No |                                                        Lesson                                                         |                                                                                                                                  Notes                                                                                                                                   |                         Link/Source                          |
|:--:|:---------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------:|
| 1 |                                                      Introduction                                                      |                                                                                                                Basics of the Model Optimizer,  Optimization techniques & impact, Supported Frameworks, Custom layers                                                     |                       ------/------                          |
| 2 |                                                  The Model Optimizer                                                   |                                                                                                                The model optimizer process, Local Configuration                                                                                                          |                            [10]                              |
| 3 |                                                Optimization Techniques                                                 |                                                                                                                      Quantization, Freezing, Fusion                                                                                                                      |                         [11 - 12]                            |
| 4 |                                                 Supported Frameworks                                                   |                                                                                                                  Caffe, TensorFlow, MXNet, ONNX, Kaldi                                                                                                                   |                         [13 - 17]                            |
| 5 |                                              Intermediate Representations                                              |                                                                                                                 OpenVINO™ Toolkit’s standard structure and naming for neural network architectures, XML file and a binary file                                           |                         [18 - 20]                            |
| 6 |                                     Using the Model Optimizer with TensorFlow Models                                   |                                                                                                                Using the Model Optimizer with TensorFlow Models                                                                                                          |                         [21 - 22]                            |
| 7 |                                              Exercise: Convert a TF Model                                              |                                                                                                                    Excercise on convert a tf model                                                                                                                       |                       ------/------                          |
| 8 |                                              Solution: Convert a TF Model                                              |                                                                                                                     Solution of convert a tf model                                                                                                                       |                       ------/------                          |
| 9 |                                     Using the Model Optimizer with Caffe Models                                        |                                                                                                                Nothing about freezing the model, Need to feed both the .caffemodel file, as well as a .prototxt file                                                     |                            [23]                              |
|10 |                                           Exercise: Convert a Caffe Model                                              |                                                                                                                    Excercise on convert a caffe model                                                                                                                    |                       ------/------                          |
|11 |                                           Solution: Convert a Caffe Model                                              |                                                                                                                     Solution of convert a caffe model                                                                                                                    |                       ------/------                          |
|12 |                                     Using the Model Optimizer with ONNX Models                                         |                                                                                                                    Model Optimizer with ONNX Models, PyTorch to ONNX                                                                                                     |                         [24 - 26]                            |
|13 |                                          Exercise: Convert an ONNX Model                                               |                                                                                                                   Excercise on covert an ONNX model                                                                                                                      |                       ------/------                          |
|14 |                                          Solution: Convert an ONNX Model                                               |                                                                                                                   Solution of convert and ONNX model                                                                                                                     |                       ------/------                          |
|15 |                                             Cutting Parts of a Model                                                   |                                                                                                                Mostly applicable for TensorFlow models, Two main command line arguments to use for cutting a model: --input and --output                                 |                            [27]                              |
|16 |                                                Supported Layers                                                        |                                                                                                                     Supported and unsupported layers                                                                                                                     |                            [28]                              |
|17 |                                                  Custom Layers                                                         |                                                                                                                       Register the custom layers                                                                                                                         |                          [29 - 30]                           |
|18 |                                            Exercise: Custom Layers                                                     |                                                                                                                Example Custom Layer: The Hyperbolic Cosine (cosh) Function                                                                                               |                       ------/------                          |
|19 |                                                     Recap                                                              |                                                                                                                Basics of the Model Optimizer,  Optimization techniques & impact, Supported Frameworks, Custom layers                                                     |                       ------/------                          |
|20 |                                                 Lesson Glossary                                                        |                                                                                                                 Short note of the lesson                                                                                                                                 |                       ------/------                          |


#### Lesson-4: The Inference Engine
| No |                                                        Lesson                                                         |                                                                                                                                  Notes                                                                                                                                   |                         Link/Source                          |
|:--:|:---------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------:|
| 1 |                                                      Introduction                                                      |                                                                                                                Inference Engine, Supported device, Feeding an Intermediate Representation to the Inference Engine, Making Inference Requests, Handling Results           |                       ------/------                          |
| 2 |                                                  The Inference Engine                                                  |                                                                                                                Runs the actual inference on a model, Only works with the Intermediate Representations                                                                    |                            [31]                              |
| 3 |                                                   Supported Devices                                                    |                                                                                                                CPUs, including integrated graphics processors, GPUs, FPGAs, and VPUs                                                                                     |                         [32 - 33]                            |
| 4 |                                        Using the Inference Engine with an IR                                           |                                                                                                                 IECore, IENetwork, Check Supported Layers, CPU extension                                                                                                 |                         [34 - 36]                            |
| 5 |                                    Exercise: Feed an IR to the Inference Engine                                        |                                                                                                                 Exercise on feed an IR to the inference engine                                                                                                           |                       ------/------                          |
| 6 |                                    Solution: Feed an IR to the Inference Engine                                        |                                                                                                                Solution of feed an IR to the inference engine                                                                                                            |                       ------/------                          |
| 7 |                                      Sending Inference Requests to the IE                                              |                                                                                                                ExecutableNetwork, Two types of inference requests: Synchronous and Asynchronous                                                                          |                         [37 - 38]                            |
| 8 |                                            Asynchronous Requests                                                       |                                                                                                                Synchronous: only one frame is being processed at once, Asynchronous: other tasks may continue while waiting on the IE to respond                         |                         [39 - 41]                            |
| 9 |                                       Exercise: Inference Requests                                                     |                                                                                                                 Inference requests (asynchronous & synchronous) excercise                                                                                                |                       ------/------                          |
|10 |                                        Solution: Inference Requests                                                    |                                                                                                                      Synchronous and Asynchronous Solution                                                                                                               |                       ------/------                          |
|11 |                                             Handling Results                                                           |                                                                                                                 InferRequest attributes - namely, inputs, outputs and latency                                                                                            |                            [42]                              |
|12 |                                        Integrating into Your App                                                       |                                                                                                                  Adding some further customization to your app                                                                                                           |                         [43 - 45]                            |
|13 |                                         Exercise: Integrate into an App                                                |                                                                                                                   Excercise on integrate into an App                                                                                                                     |                       ------/------                          |
|14 |                                         Solution: Integrate into an App                                                |                                                                                                                   Solution of integrate into an App                                                                                                                      |                       ------/------                          |
|15 |                                     Behind the Scenes of Inference Engine                                              |                                                                                                                Inference Engine is built and optimized in C++, The exact optimizations differ by device with the Inference Engine                                        |                          [46 - 47]                           |
|16 |                                                  Recap                                                                 |                                                                                                                Inference Engine, Supported device, Feeding an Intermediate Representation to the Inference Engine, Making Inference Requests, Handling Results           |                       ------/------                          |
|17 |                                              Lesson Glossary                                                           |                                                                                                                    Short note of the lesson                                                                                                                              |                       ------/------                          |


#### Lesson-5: Deploying an Edge App
| No |                                                        Lesson                                                         |                                                                                                                                  Notes                                                                                                                                   |                         Link/Source                          |
|:--:|:---------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------:|
| 1 |                                                      Introduction                                                      |                                                                                                                OpenCV, Input Streams in OpenCV, Processing Model Outputs for Additional Useful Information, MQTT and their use with IoT devices, Performance basics      |                       ------/------                          |
| 2 |                                                      OpenCV Basics                                                     |                                                                                                                Uses of OpenCV, Useful OpenCV function: VideoCapture, resize, cvtColor, rectangle, imwrite                                                                |                            [48]                              |
| 3 |                                                Handling Input Streams                                                  |                                                                                                                    Open & Read A Video, Closing the Capture                                                                                                              |                       ------/------                          |
| 4 |                                          Exercise: Handling Input Streams                                              |                                                                                                                Handle image, video or webcam, resize, Add Canny Edge Detection to the frame, Write out the frame                                                         |                       ------/------                          |
| 5 |                                          Solution: Handling Input Streams                                              |                                                                                                                Handle image, video or webcam, resize, Add Canny Edge Detection to the frame, Write out the frame                                                         |                       ------/------                          |
| 6 |                                    Gathering Useful Information from Model Outputs                                     |                                                                                                                Information from one model could even be further used in an additional model                                                                              |                       ------/------                          |
| 7 |                                          Exercise: Process Model Outputs                                               |                                                                                                                   Excercise on model outputs process                                                                                                                     |                       ------/------                          |
| 8 |                                          Solution: Process Model Outputs                                               |                                                                                                                    Solution of model outputs process                                                                                                                     |                       ------/------                          |
| 9 |                                                 Intro to MQTT                                                          |                                                                                                                Stands for MQ Telemetry Transport, Lightweight publish/subscribe architecture, Port 1883                                                                  |                         [49 - 50]                            |
|10 |                                             Communicating with MQTT                                                    |                                                                                                                MQTT Python library: paho-mqtt, Publishing or subscribing parameters                                                                                      |                         [51 - 52]                            |
|11 |                                            Streaming Images to a Server                                                |                                                                                                                FFmpeg (“fast forward” MPEG), Setting up FFmpeg, Sending frames to FFmpeg                                                                                 |                         [53 - 55]                            |
|12 |                                Handling Statistics and Images from a Node Server                                       |                                                                                                                Node server can be used to handle the data coming in from the MQTT and FFmpeg servers                                                                     |                            [56]                              |
|13 |                                         Exercise: Server Communications                                                |                                                                                                                   Excercise on server communication using node.js and mqtt                                                                                               |                       ------/------                          |
|14 |                                         Solution: Server Communications                                                |                                                                                                                   Solution of server communication using node.js and mqtt                                                                                                |                       ------/------                          |
|15 |                                          Analyzing Performance Basics                                                  |                                                                                                                Not to skip past the accuracy of your edge AI model, Lighter, quicker models, Lower precision                                                             |                            [57]                              |
|16 |                                               Model Use Cases                                                          |                                                                                                                   Figure out additional use cases for a given model or application                                                                                       |                            [58]                              |
|17 |                                          Concerning End User Needs                                                     |                                                                                                                    Consider the project needs                                                                                                                            |                       ------/------                          |
|18 |                                                     Recap                                                              |                                                                                                                OpenCV, Input Streams in OpenCV, Processing Model Outputs for Additional Useful Information, MQTT and their use with IoT devices, Performance basics      |                       ------/------                          |
|19 |                                               Lesson Glossary                                                          |                                                                                                                          Short note of the lesson                                                                                                                        |                       ------/------                          |
|20 |                                                 Course Recap                                                           |                                                                                                                Basics of AI at the Edge, Pre-trained models, the Model Optimizer, Inference Engine, Deploying an app at the edge                                         |                            [59]                              |
|21 |                                             Partner with Intel                                                         |                                                                                                                 Benefits of partner with Intel                                                                                                                           |                       ------/------                          |


## Resources
* [1] [A guide to internet of things infographics](https://www.intel.com/content/www/us/en/internet-of-things/infographics/guide-to-iot.html)
* [2] [Intel® Distribution of OpenVINO™ Toolkit](https://software.intel.com/en-us/openvino-toolkit)
* [3] [Hardware requirements for the Intel® Distribution of OpenVINO](https://software.intel.com/en-us/openvino-toolkit/hardware)
* [4] [Set-up like a Raspberry Pi with an Intel® Neural Compute Stick 2](https://software.intel.com/en-us/articles/intel-neural-compute-stick-2-and-open-source-openvino-toolkit)
* [5] [Intel® DevCloud platform](https://software.intel.com/en-us/devcloud/edge)
* [6] [Pretrained Models](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models)
* [7] [Image Classification vs. Object Detection vs. Image Segmentation](https://medium.com/analytics-vidhya/image-classification-vs-object-detection-vs-image-segmentation-f36db85fe81)
* [8] [Understanding SSD MultiBox — Real-Time Object Detection In Deep Learning](https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab)
* [9] [Going beyond the bounding box with semantic segmentation](https://thegradient.pub/semantic-segmentation/)
* [10] [Model Optimizer Developer Guide](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [11] [Quantization](https://nervanasystems.github.io/distiller/quantization.html)
* [12] [Model Optimization Techniques](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Model_Optimization_Techniques.html)
* [13] [Caffe](https://caffe.berkeleyvision.org/)
* [14] [TensorFlow](https://www.tensorflow.org/)
* [15] [MXNet](https://mxnet.apache.org/)
* [16] [ONNX](https://onnx.ai/)
* [17] [Kaldi](https://kaldi-asr.org/doc/dnn.html)
* [18] [Converting a Model to Intermediate Representation (IR)](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_convert_model_Converting_Model.html)
* [19] [Supported Framework Layers](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html)
* [20] [Intermediate Representation Notation Reference Catalog](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_convert_model_IRLayersCatalogSpec.html)
* [21] [Converting a TensorFlow* Model](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html)
* [22] [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
* [23] [Converting a Caffe* Model](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Caffe.html)
* [24] [Converting a ONNX* Model](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX.html)
* [25] [ONNX Model Zoo](https://github.com/onnx/models)
* [26] [Converting a PyTorch model using ONNX](https://michhar.github.io/convert-pytorch-onnx/)
* [27] [Cutting Off Parts of a Model](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_convert_model_Cutting_Model.html)
* [28] [Supported Framework Layers](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html)
* [29] [Custom Layers in the Model Optimizer](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer.html)
* [30] [Offloading Sub-Graph Inference to TensorFlow*](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_customize_model_optimizer_Offloading_Sub_Graph_Inference.html)
* [31] [Inference Engine Developer Guide](https://docs.openvinotoolkit.org/2019_R3/_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html)
* [32] [Supported Devices](https://docs.openvinotoolkit.org/2019_R3/_docs_IE_DG_supported_plugins_Supported_Devices.html)
* [33] [Distribution of OpenVINO™ Toolkit on Raspberry Pi*](https://software.intel.com/en-us/articles/model-downloader-optimizer-for-openvino-on-raspberry-pi)
* [34] [IE Core](https://docs.openvinotoolkit.org/2019_R3/classie__api_1_1IECore.html)
* [35] [IE Network](https://docs.openvinotoolkit.org/2019_R3/classie__api_1_1IENetwork.html)
* [36] [IE Python API](https://docs.openvinotoolkit.org/2019_R3/ie_python_api.html)
* [37] [Executable Network documentation](https://docs.openvinotoolkit.org/2019_R3/classie__api_1_1ExecutableNetwork.html)
* [38] [Infer Request documentation](https://docs.openvinotoolkit.org/2019_R3/classie__api_1_1InferRequest.html)
* [39] [synchronous/asynchronous API](https://whatis.techtarget.com/definition/synchronous-asynchronous-API)
* [40] [Integrate the Inference Engine with Your Application](https://docs.openvinotoolkit.org/2019_R3/_docs_IE_DG_Integrate_with_customer_application_new_API.html)
* [41] [Object Detection SSD C++ Demo, Async API Performance Showcase](https://github.com/opencv/open_model_zoo/blob/master/demos/object_detection_demo_ssd_async/README.md)
* [42] [InferenceEngine::Blob Class Reference](https://docs.openvinotoolkit.org/2019_R3/classInferenceEngine_1_1Blob.html)
* [43] [Intel®’s IoT Apps Across Industries](https://www.intel.com/content/www/us/en/internet-of-things/industry-solutions.html)
* [44] [Starting Your First IoT Project](https://hackernoon.com/the-ultimate-guide-to-starting-your-first-iot-project-8b0644fbbe6d)
* [45] [OpenVINO™ on a Raspberry Pi and Intel® Neural Compute Stick](https://www.pyimagesearch.com/2019/04/08/openvino-opencv-and-movidius-ncs-on-the-raspberry-pi/)
* [46] [What is the best programming language for Machine Learning?](https://towardsdatascience.com/what-is-the-best-programming-language-for-machine-learning-a745c156d6b7)
* [47] [Optimization Guide](https://docs.openvinotoolkit.org/2019_R3/_docs_optimization_guide_dldt_optimization_guide.html)
* [48] [OpenCV tutorial](https://docs.opencv.org/master/d9/df8/tutorial_root.html)
* [49] [MQTT](http://mqtt.org/)
* [50] [Basics of MQTT](https://internetofthingsagenda.techtarget.com/definition/MQTT-MQ-Telemetry-Transport)
* [51] [paho-mqtt - PyPI](https://pypi.org/project/paho-mqtt/)
* [52] [Developer Kits for IoT - Intel® Software](https://software.intel.com/en-us/iot/hardware/all)
* [53] [FFMPEG](https://www.ffmpeg.org/)
* [54] [Create your own video streaming server with Linux](https://opensource.com/article/19/1/basic-live-video-streaming-server)
* [55] [OpenCV – Stream video to web browser/HTML page](https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/)
* [56] [Node.js](https://nodejs.org/en/about/)
* [57] [Introduction to the Performance Topics](https://docs.openvinotoolkit.org/2019_R3/_docs_IE_DG_Intro_to_Performance.html)
* [58] [Deep Learning for Distracted Driving Detection](https://www.nauto.com/blog/nauto-engineering-deep-learning-for-distracted-driver-monitoring)
* [59] [Intel® DevMesh](https://devmesh.intel.com/)

## Certification the Program
![Intel Scholarship Winner Badge](Capture.PNG)

