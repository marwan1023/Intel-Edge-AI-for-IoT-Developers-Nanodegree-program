# Smart Queuing System
The goal of this project is building an application to reduce congestion and queuing systems.
## The Scenarios
1. [Manufacturing Sector](https://github.com/RoumaissaaMadoui/intel-edge-AI-for-IoT-developers-nanodegree-projects/blob/master/smart-queuing-system/Scenarios/Scenario%201.pdf)
2. [Retail Sector](https://github.com/RoumaissaaMadoui/intel-edge-AI-for-IoT-developers-nanodegree-projects/blob/master/smart-queuing-system/Scenarios/Scenario%202.pdf)
3. [Transportation Sector](https://github.com/RoumaissaaMadoui/intel-edge-AI-for-IoT-developers-nanodegree-projects/blob/master/smart-queuing-system/Scenarios/Scenario%203.pdf)
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
   | ![Manufacturing_cpu](./results_gif/m_cpu.gif) | ![Manufacturing_fpga](./results_gif/m_fpga.gif) | ![Manufacturing_gpu](./results_gif/m_gpu.gif) | ![Manufacturing_vpu](./results_gif/m_vpu.gif) |

2. Retail Sector

   |  CPU   |  FPGA |  GPU  |  VPU  |
   |  :---: | :---: | :---: | :---: |     
   | ![Retail_cpu](./results_gif/r_cpu.gif) | ![Retail_fpga](./results_gif/r_fpga.gif) | ![Retail_gpu](./results_gif/r_gpu.gif) | ![Retail_vpu](./results_gif/r_vpu.gif) |

3. Transportation Sector

   |  CPU   |  FPGA |  GPU  |  VPU  |
   |  :---: | :---: | :---: | :---: |      
   | ![Transportation_cpu](./results_gif/t_cpu.gif) | ![Transportation_fpga](./results_gif/t_fpga.gif) | ![Transportation_gpu](./results_gif/t_gpu.gif) | ![Transportation_vpu](./results_gif/t_vpu.gif) |
