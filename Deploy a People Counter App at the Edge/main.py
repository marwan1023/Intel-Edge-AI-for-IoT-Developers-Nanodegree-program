
"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import time
import cv2
import sys
import numpy as np
import socket
import json
import paho.mqtt.client as mqtt
from random import randint
from inference import Network
from argparse import ArgumentParser

INPUT_STREAM = "/home/workspace/resources/Pedestrian_Detect_2_1_1.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
ADAS_MODEL = "/home/workspace/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml"

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-pc", "--perf_counts", type=str, default=False,
                        help="Print performance counters")
    return parser


def ssd_out(frame, result,args, width, height):
    count = 0     
    classid = 0
    for obj in result[0][0]:  
        confidence = obj[2]  
        # if id == 1, it is person 
        classid = int(obj[1]) 
        if classid == 1: 
            if confidence >= args.prob_threshold: 
                xmin = int(obj[3] * width)
                ymin = int(obj[4] * height)
                xmax = int(obj[5] * width)
                ymax = int(obj[6] * height)  
                #yellow rgb is 255,255,0, but in opencv bgr  
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                Person_confidence = '%s: %.1f%%' % ("Person", round(confidence * 100, 1))
                cv2.putText(frame, Person_confidence, (10, 90), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
                count += 1    
                
    return frame, count


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold 
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()  
    ### TODO: Handle the input stream ###
    image_flag = False
    # If input is CAM 
    if args.input == 'CAM':
        input_stream = 0 
    # If input is image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        image_flag = True
        input_stream = args.input
    # input is video file
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "file doesn't exist"

    if input_stream and not image_flag: 
        cap = cv2.VideoCapture(args.input)
        cap.open(args.input) 
        
    width = int(cap.get(3))
    height = int(cap.get(4))
    # Process frames until the video ends, or process is exited
    counter = 0
    duration = 0
    total_count = 0
    current_count = 0 
    total_inference_time = 0
    last_count = 0 
    threshold_value = 2  
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read() 
        # FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        # fpss = FPS().start() 
        # frame counter
        counter += 1
        if not flag:
            break
        key_pressed = cv2.waitKey(60) 
        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape) 
        # calculating time for the performance in different models
        infer_timer = time.time()
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(p_frame)  
        ### TODO: Wait for the result ### 
        if infer_network.wait() == 0:
            # calculating time for the performance in different models
            inferece_time = time.time() - infer_timer  
            ### TODO: Get the results of the inference request ###  
            result = infer_network.get_output() 
            ### TODO: Extract any desired stats from the results ### 
            frame, count = ssd_out(frame, result, args, width, height)   
            
            ### TODO: Calculate and send relevant information on ###   
            current_count = count    
            if current_count > last_count:
                # if someone enter frame, time start counting
                start_time = time.time() 
                total_count = total_count + current_count - last_count 
                
            # Person duration in the video is calculated
            if current_count < last_count:
                duration = int(time.time() - start_time)
                # if detection failed and double counted, decrease its value and threshold_value is 1 second
                if duration < threshold_value:
                    total_count = total_count - 1  
                if duration >= 4:
                    ### Topic "person/duration": key of "duration" ###
                    client.publish("person/duration", json.dumps({"duration": duration}))
                    ### Topic "person": keys of "count" and "total" ###
                    client.publish("person", json.dumps({"total": total_count})) 
            
            ### current_count, total_count and duration to the MQTT server ###
            client.publish("person", json.dumps({"count": count}))
            last_count = current_count
             
            if key_pressed == 27:
                break 
        # for performance message
        total_inference_time = inferece_time
        msg = "Inference time: %.3f ms" % (total_inference_time * 1000)  
        
        # fpss.update()
        # fpss.stop()
        # fpss = fpss.elapsed()
        #msg_fps = "FPS: " + str(int(fps))
        cv2.putText(frame, msg, (5, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
        ### TODO: Send the frame to the FFMPEG server ###   
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###
        if image_flag:
            cv2.imwrite('output_image.jpg', frame)
            
    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    ### Disconnect from MQTT 
    client.disconnect()   

def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)



if __name__ == '__main__':
    main()