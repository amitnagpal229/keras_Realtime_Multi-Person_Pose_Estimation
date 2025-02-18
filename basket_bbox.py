import argparse
import cv2
import time
import pickle
import numpy as np
import math


def init_video():
    # Video reader
    video_in = cv2.VideoCapture(args.video)
    width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    video_out = cv2.VideoWriter(video_out_filename, cv2.VideoWriter_fourcc(*'mp4v'), args.out_fps, (width, height))
    return video_in, video_out, width, height


def get_orig_image(previous_frame_id, frame_id):
    for i in range(previous_frame_id + 1, frame_id):
        ret, image = cam_in.read()
        cam_out.write(image)

    return cam_in.read()


def draw_bbox(input_image, bbox, color=None):
    (x1, y1), (x2, y2) = bbox[:4]
    cv2.rectangle(input_image, (x1, y1), (x2, y2), color, 3)
    return input_image


def process():
    previous_frame_id = -1  # Previous video frame processed
    for bbox_frame_id in sorted(basket_bbox.keys()):
        video_frame_id = int((bbox_frame_id - args.frame_id_offset)/args.frame_ratio)
        ret_val, orig_image = get_orig_image(previous_frame_id, video_frame_id)
        previous_frame_id = video_frame_id
        bbox = basket_bbox[bbox_frame_id]
        orig_image = draw_bbox(orig_image, bbox, [255, 255, 0])
        cam_out.write(orig_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--basket_bbox', type=str, required=True, help='pkl file with basket bbox')
    parser.add_argument('--video', type=str, required=True, help='video clip')
    parser.add_argument('--out_fps', type=int, required=True, help='fps of output video')
    parser.add_argument('--frame_id_offset', type=int, required=True, help='frame_id of first frame in video')
    parser.add_argument('--frame_ratio', type=int, default=3, help='frame ratio between video file & bbox pkl file')

    args = parser.parse_args()
    video_out_filename = args.video.rsplit(".", 1)[0].rsplit("_", 1)[0] + "_trackpose_basket.mp4"

    # Pickle files for pose and tracking
    basket_bbox = pickle.load(open(args.basket_bbox, "rb"), fix_imports=True)

    # init input and output videos
    cam_in, cam_out, im_width, im_height = init_video()

    process()
