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

<img src="/Computer Pointer Controller/starter/Images/pipeline.png" />

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
<img src="/Computer Pointer Controller/starter/Images/cpu/asycncpu1.png" />   | <img src="/Computer Pointer Controller/starter/Images/cpu/asycncpu2.png" />
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

And more sources, see this link. [The Free Foundation course is from Udacity and Intel](https://github.com/marwan1023/Udacity-Intel-Edge-AI-for-IoT-Developers-Nanodegree)|
[Intel® Edge AI Fundamentals with OpenVINO™](https://www.udacity.com/course/intel-edge-AI-fundamentals-with-openvino--ud132)


## Certification the Program
![Intel Scholarship Winner Badge](Capture.PNG)

