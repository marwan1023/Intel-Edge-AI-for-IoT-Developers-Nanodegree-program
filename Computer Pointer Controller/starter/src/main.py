import cv2
import os
import logging
import time
import numpy as np
from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection_model import FaceDetectionModel
from landmark_detection_model import LandmarkDetectionModel
from head_pose_estimation_model import HeadPoseEstimationModel
from gaze_estimation_model import GazeEstimationModel
from argparse import ArgumentParser


def build_argparser():
    """
    parse commandline argument
    return ArgumentParser object
    """
    parser = ArgumentParser()
    parser.add_argument("-fdm", "--faceDetectionModel", type=str, required=True,
                        help="Specify path of xml file of face detection model")

    parser.add_argument("-lrm", "--landmarkRegressionModel", type=str, required=True,
                        help="Specify path of xml file of landmark regression model")

    parser.add_argument("-hpm", "--headPoseEstimationModel", type=str, required=True,
                        help="Specify path of xml file of Head Pose Estimation model")

    parser.add_argument("-gem", "--gazeEstimationModel", type=str, required=True,
                        help="Specify path of xml file of Gaze Estimation model")

    parser.add_argument("-inp", "--input", type=str, required=True,
                        help="Specify path of input Video file or cam for webcam")

    parser.add_argument("-flags", "--previewFlags", required=False, nargs='+',
                        default=[],
                        help="Specify flag from ff, fl like -flags ff fl(Space separated if multiple values)"
                             "face_frame for faceDetectionModel, face_eyes for landmarkRegressionModel")

    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Specify probability threshold for face detection model")

    parser.add_argument("-d", "--device", required=False, type=str, default='CPU',
                        help="Specify Device for inference"
                             "It can be CPU, GPU, FPGU, MYRID")
    parser.add_argument("-o", '--output_path', default='/outcomes/', type=str)
    return parser


def draw_preview(eye_cords,left_eye_image, right_eye_image,gaze_vector, cropped_image,face_cords,preview_flags,frame):

    preview_frame = frame.copy()


    if 'face_frame' in preview_flags:
        if len(preview_flags) != 1:
            preview_frame = cropped_image
        cv2.rectangle(frame, (face_cords[0][0], face_cords[0][1]), (face_cords[0][2], face_cords[0][3]),
                      (255,0,0),2)

    if 'face_eyes' in preview_flags:
        cv2.rectangle(cropped_image, (eye_cords[0][0]-10, eye_cords[0][1]-10), (eye_cords[0][2]+10, eye_cords[0][3]+10),
                      (0,255,0),2)
        cv2.rectangle(cropped_image, (eye_cords[1][0]-10, eye_cords[1][1]-10), (eye_cords[1][2]+10, eye_cords[1][3]+10),
                      (0,255,0),2)

        return preview_frame


def main():
    args = build_argparser().parse_args()
    logger = logging.getLogger('main')

    is_benchmarking = False
    # initialize variables with the input arguments for easy access
    model_path_dict = {
        'FaceDetectionModel': args.faceDetectionModel,
        'LandmarkRegressionModel': args.landmarkRegressionModel,
        'HeadPoseEstimationModel': args.headPoseEstimationModel,
        'GazeEstimationModel': args.gazeEstimationModel
    }
    preview_flags = args.previewFlags
    input_filename = args.input
    device_name = args.device
    prob_threshold = args.prob_threshold
    output_path = args.output_path

    if input_filename.lower() == 'cam':
        feeder = InputFeeder(input_type='cam')
    else:
        if not os.path.isfile(input_filename):
            logger.error("Unable to find specified video file")
            exit(1)
        feeder = InputFeeder(input_type='video', input_file=input_filename)

    for model_path in list(model_path_dict.values()):
        if not os.path.isfile(model_path):
            logger.error("Unable to find specified model file" + str(model_path))
            exit(1)

    # instantiate model
    face_detection_model = FaceDetectionModel(model_path_dict['FaceDetectionModel'], device_name, threshold=prob_threshold)
    landmark_detection_model = LandmarkDetectionModel(model_path_dict['LandmarkRegressionModel'], device_name, threshold=prob_threshold)
    head_pose_estimation_model = HeadPoseEstimationModel(model_path_dict['HeadPoseEstimationModel'], device_name, threshold=prob_threshold)
    gaze_estimation_model = GazeEstimationModel(model_path_dict['GazeEstimationModel'], device_name, threshold=prob_threshold)

    if not is_benchmarking:
        mouse_controller = MouseController('medium', 'fast')

    # load Models
    start_model_load_time = time.time()
    face_detection_model.load_model()
    landmark_detection_model.load_model()
    head_pose_estimation_model.load_model()
    gaze_estimation_model.load_model()
    total_model_load_time = time.time() - start_model_load_time

    feeder.load_data()

    out_video = cv2.VideoWriter(os.path.join('output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), int(feeder.get_fps()/10),
                                (1920, 1080), True)

    frame_count = 0
    start_inference_time = time.time()
    for ret, frame in feeder.next_batch():

        if not ret:
            break

        frame_count += 1

        key = cv2.waitKey(60)

        try:
            face_cords, cropped_image = face_detection_model.predict(frame)

            if type(cropped_image) == int:
                logger.warning("Unable to detect the face")
                if key == 27:
                    break
                continue

            left_eye_image, right_eye_image, eye_cords = landmark_detection_model.predict(cropped_image)
            pose_output = head_pose_estimation_model.predict(cropped_image)
            mouse_cord, gaze_vector = gaze_estimation_model.predict(left_eye_image, right_eye_image, pose_output)

        except Exception as e:
            logger.warning("Could predict using model" + str(e) + " for frame " + str(frame_count))
            continue

        image = cv2.resize(frame, (500, 500))

        if not len(preview_flags) == 0:
            preview_frame = draw_preview(eye_cords,left_eye_image, right_eye_image,gaze_vector, cropped_image,face_cords,preview_flags,frame)
            image = np.hstack((cv2.resize(frame, (500, 500)), cv2.resize(preview_frame, (500, 500))))

        cv2.imshow('preview', image)
        out_video.write(frame)

        if frame_count % 5 == 0 and not is_benchmarking:
            mouse_controller.move(mouse_cord[0], mouse_cord[1])

        if key == 27:
            break

    total_time = time.time() - start_inference_time
    total_inference_time = round(total_time, 1)
    fps = frame_count / total_inference_time

    try:
        os.mkdir(output_path)
    except OSError as error:
        logger.error(error)

    with open(output_path+'stats.txt', 'w') as f:
        f.write(str(total_inference_time) + '\n')
        f.write(str(total_model_load_time) + '\n')
        f.write(str(fps) + '\n')


    logger.info('Model load time: ' + str(total_model_load_time))
    logger.info('Inference time: ' + str(total_inference_time))
    logger.info('FPS: ' + str(fps))

    print("Total load time:", total_time, "s")
    print("Total iinference time:", total_inference_time, "s")
    print("FPS:", fps, "frames/s")

    logger.info('Video stream ended')
    cv2.destroyAllWindows()
    feeder.close()


if __name__ == '__main__':
    main()
