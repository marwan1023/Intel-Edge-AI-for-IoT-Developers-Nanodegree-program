# Smart Queuing System
The goal of this project is building an application to reduce congestion and queuing systems.
## The Scenarios
1. [Manufacturing Sector](https:///marwan1023-Intel-Edge-AI-for-IoT-Developers-Nanodegree-program--to-confirm./blob/master/Smart%20Queue%20Monitoring%20System/Scenarios/Scenario%201.pdf)
2. [Retail Sector](https://https://github.com/marwan1023/marwan1023-Intel-Edge-AI-for-IoT-Developers-Nanodegree-program--to-confirm./blob/master/Smart%20Queue%20Monitoring%20System/Scenarios/Scenario%202.pdf)
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
   | ![Manufacturing_cpu](./results_gif/manufacturing/cpu/ezgif.com-video-to-gif_cpu.gif) | ![Manufacturing_fpga](./results_gif/manufacturing/fpga/ezgif.com-video-to-gif_Fpga.gif) | ![Manufacturing_gpu](./results_gif/manufacturing/gpu/ezgif.com-video-to-gif_gpu.gif) | ![Manufacturing_vpu](./results_gif/manufacturing/vpu/ezgif.com-video-to-gif_vpu.gif) |

2. Retail Sector

   |  CPU   |  FPGA |  GPU  |  VPU  |
   |  :---: | :---: | :---: | :---: |     
   | ![Retail_cpu](./results_gif/retail/cpu/ezgif.com-video-to-gif_cpu.gif) | ![Retail_fpga](./results_gif/retail/fpga/ezgif.com-video-to-gif_fpga.gif) | ![Retail_gpu](./results_gif/retail/gpu/ezgif.com-video-to-gif_gpu.gif) | ![Retail_vpu](./results_gif/retail/vpu/ezgif.com-video-to-gif_vpu.gif) |

3. Transportation Sector

   |  CPU   |  FPGA |  GPU  |  VPU  |
   |  :---: | :---: | :---: | :---: |      
   | ![Transportation_cpu](./results_gif/transportation/cpu/ezgif.com-video-to-gif_cpu.gif) | ![Transportation_fpga](./results_gif/transportation/fpga/ezgif.com-video-to-gif_fpga.gif) | ![Transportation_gpu](./results_gif/transportation/gpu/ezgif.com-video-to-gif_gpu.gif) | ![Transportation_vpu](./results_gif/transportation/vpu/ezgif.com-video-to-gif_vpu.gif) |
