import cv2
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='input video file name')
    args = parser.parse_args()

    cam = cv2.VideoCapture(args.video)
    ret_val, orig_image = cam.read()
    print(f"image.shape={orig_image.shape}")

