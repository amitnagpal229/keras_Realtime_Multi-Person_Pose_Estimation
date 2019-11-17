import os
import sys
import argparse
import cv2
import time
import pickle
from config_reader import config_reader

from processing2 import extract_parts, draw, get_model_heat_paf

from model.cmu_model import get_testing_model

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

currentDT = time.localtime()
start_datetime = time.strftime("-%m-%d-%H-%M-%S", currentDT)


def generate_heat_paf(in_video_file, ending_frame):
    # Video reader
    cam = cv2.VideoCapture(in_video_file)
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    ret_val, orig_image = cam.read()
    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    if ending_frame is None:
        ending_frame = video_length

    heat_paf = {}
    i = 0  # default is 0
    while (cam.isOpened()) and ret_val is True and i < ending_frame:
        if i % frame_rate_ratio == 0:
            input_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)

            tic = time.time()
            heat_paf[i] = get_model_heat_paf(input_image, params, model, model_params)
            toc = time.time()
            print(f"Generate heatmap paf, frame: {i}, took {toc - tic} seconds")

        ret_val, orig_image = cam.read()
        i += 1
    return heat_paf


def generate_pose(heat_paf):
    pose = {}
    for frame in heat_paf:
        tic = time.time()
        pose[frame] = extract_parts(heat_paf[frame][0], params, heat_paf[frame][1], heat_paf[frame][2])
        toc = time.time()
        print(f"Generate pose, frame: {frame}, took {toc - tic} seconds")
    return pose


def save_output_video(in_video_file, out_video_file, pose, ending_frame):
    # Video reader
    cam = cv2.VideoCapture(in_video_file)
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    ret_val, orig_image = cam.read()
    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    if ending_frame is None:
        ending_frame = video_length

    # Video writer
    output_fps = input_fps / frame_rate_ratio
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_file, fourcc, output_fps, (orig_image.shape[1], orig_image.shape[0]))

    i = 0
    while (cam.isOpened()) and ret_val is True and i < ending_frame:
        if i in pose:
            tic = time.time()
            canvas = draw(orig_image, pose[i][0], pose[i][1], pose[i][2])
            out.write(canvas)
            toc = time.time()
            print(f"Save output video, frame: {i}, took {toc - tic} seconds")

        ret_val, orig_image = cam.read()
        i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--video', type=str, required=True, help='input video file name')
    parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')
    parser.add_argument('--frame_ratio', type=int, default=1, help='analyze every [n] frames')
    parser.add_argument('--process_speed', type=int, default=4,
                        help='Int 1 (fastest, lowest quality) to 4 (slowest, highest quality)')
    parser.add_argument('--end', type=int, default=None, help='Last video frame to analyze')
    parser.add_argument('--generate_heat_paf', type=str, help='video file to process')
    parser.add_argument('--generate_pose', type=str, default=None, help='provide heat map and paf pkl file')
    parser.add_argument('--save_output_video', type=str, default=None,
                        help='provide input video file. assumes presence of pose pkl file')

    args = parser.parse_args()
    keras_weights_file = args.model
    frame_rate_ratio = args.frame_ratio
    process_speed = args.process_speed
    ending_frame = args.end

    # load config
    params, model_params = config_reader()

    print('start processing...')
    if args.generate_heat_paf is not None:
        # Video input
        video_file = args.generate_heat_paf
        hpaf_file = video_file.rsplit(".", 1)[0] + "_hpaf.pkl"

        # load model
        # authors of original model don't use
        # vgg normalization (subtracting mean) on input images
        model = get_testing_model()
        model.load_weights(keras_weights_file)

        # Video reader
        cam = cv2.VideoCapture(video_file)
        input_fps = cam.get(cv2.CAP_PROP_FPS)
        ret_val, orig_image = cam.read()
        video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

        if ending_frame is None:
            ending_frame = video_length

        scale_search = [1, .5, 1.5, 2]  # [.5, 1, 1.5, 2]
        scale_search = scale_search[0:process_speed]

        params['scale_search'] = scale_search

        heat_paf = generate_heat_paf(video_file, ending_frame)
        pickle.dump(heat_paf, open(hpaf_file, "wb"))

    if args.generate_pose is not None:
        hpaf_file = args.generate_pose
        pose_file = hpaf_file.rsplit("_", 1)[0] + "_pose.pkl"

        hpaf = pickle.load(open(hpaf_file, "rb"))
        pose = generate_pose(hpaf)
        pickle.dump(pose, open(pose_file, "wb"))

    if args.save_output_video is not None:
        video_file = args.save_output_video
        pose_file = video_file.rsplit(".", 1)[0] + "_pose.pkl"
        output_file = video_file.rsplit(".", 1)[0] + "_output.mp4"

        pose = pickle.load(open(pose_file, "rb"))
        save_output_video(video_file, output_file, pose, ending_frame)
