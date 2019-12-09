import argparse
import cv2
import time
import pickle
import numpy as np
import math
import json

from action import limb_str_to_id
from util import colors

limb_id_to_str = ["rshoulder", "lshoulder", "rupperarm", "rlowerarm", "lupperarm", "llowerarm", "rhipneck",
                  "rupperleg", "rlowerleg", "lhipneck", "lupperleg", "llowerleg", "noseneck", "reyenose", "reyeear",
                  "leyenose", "leyeear"]

# def is_arm_up(pose_array):
#     limb_id_to_str[rlowerarm]
# def jumped(track):
#     max_pose_point = im_height
#     max_arm_angle = -90
#     for limb_id, limb in pose.items():
#         max_pose_point = min(max_pose_point, limb['from'][1], limb['to'][1])
#         if limb_id in ("rlowerarm", "llowerarm", "rupperarm", "lupperarm"):
#             max_arm_angle = max(max_arm_angle, limb['ang'])
#     return max_pose_point, max_arm_angle > 0, False


def hip_joint(upperleg, hipneck):
    if upperleg is not None:
        return upperleg['from']
    elif hipneck is not None:
        return hipneck['to']
    else:
        return 0, 0


def get_hip_joint(pose):
    rhip = hip_joint(pose.get('rupperleg', None), pose.get('rhipneck', None))
    lhip = hip_joint(pose.get('lupperleg', None), pose.get('lhipneck', None))
    if rhip is None and lhip is None:
        return None
    elif rhip is None and lhip is not None:
        return lhip
    elif rhip is not None and lhip is None:
        return rhip
    else:
        return int((rhip[0]+lhip[0])/2), int((rhip[1]+lhip[1])/2)


# Returns a tuple x, where:
# x[0] = y location of top joint
# x[1] = True if arms are raised
# x[2] = True if pose is sitting
def analyze_pose(pose):
    max_pose_point = 1080
    max_arm_angle = -90
    for limb_id, limb in pose.items():
        max_pose_point = min(max_pose_point, limb['from'][1], limb['to'][1])
        if limb_id in ("rlowerarm", "llowerarm", "rupperarm", "lupperarm"):
            max_arm_angle = max(max_arm_angle, limb['ang'])
    return max_pose_point, max_arm_angle > 0, False


# True if wrist or elbow (any) is above the neck
def arms_up(pose):
    neck = max(pose.get('lshoulder', {'from': (0, 0)})['from'], pose.get('rshoulder', {'from': (0, 0)})['from'])
    lelbow = min(pose.get('llowerarm', {'from': (1920, 1080)})['from'], pose.get('lupperarm', {'to': (1920, 1080)})['to'])
    relbow = min(pose.get('rlowerarm', {'from': (1920, 1080)})['from'], pose.get('rupperarm', {'to': (1920, 1080)})['to'])
    wrist = min(pose.get('llowerarm', {'to': (1920, 1080)})['to'], pose.get('rlowerarm', {'to': (1920, 1080)})['to'])
    return min(lelbow[1], relbow[1], wrist[1]) < neck[1]


def get_max_travel(history, value):
    if value == 0 or history[-1] == 0:
        return 0

    first_nonzero = np.argwhere(history == 0)
    start = 0
    if first_nonzero.shape[0] > 0:
        start = first_nonzero[-1][0] + 1

    history = history[start:]
    low = np.min(history)
    high = np.max(history)
    low_diff = abs(low-value)
    high_diff = abs(high-value)
    if low_diff >= high_diff:
        return value - low
    else:
        return value - high


def process(tracks, unmatched_poses):
    track_filter = {}
    frame_ids = sorted(unmatched_poses.keys())
    frame_offset = frame_ids[0]
    for track in tracks:
        filter = np.zeros((3, len(frame_ids)), dtype=np.int16)
        for frame, pose in tracks[track].items():
            hip = get_hip_joint(pose)
            if hip is not None:
                frame_index = int((frame - frame_offset) / 3)
                filter[:, frame_index] = hip[0], hip[1], arms_up(pose)

        mean_filter = np.zeros((3, len(frame_ids)), dtype=np.int16)
        for i in range(1, len(frame_ids)):
            history = filter[:, max(0, i - 5): i]
            mean_filter[2][i] = max(np.max(history[2]), filter[2][i])
            mean_filter[1][i] = get_max_travel(history[1], filter[1][i])
            mean_filter[0][i] = get_max_travel(history[0], filter[0][i])
        track_filter[track] = mean_filter

        neg_run = 0
        arms = 0
        for i in range(4, len(frame_ids)):
            if np.argwhere(mean_filter[1][i-4:i] < -20).shape[0] == 4 and mean_filter[2][i] >= 1:
                print(f"Track: {track}, jumped around frame: {i}")

    return track_filter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_track_pkl_file', type=str, required=True, help='pose track pkl file name')

    args = parser.parse_args()
    track_file = args.pose_track_pkl_file
    out_file = track_file.rsplit(".", 1)[0] + "_analysis.pkl"

    tracks, poses = pickle.load(open(track_file, "rb"))
    analysis = process(tracks, poses)

    pickle.dump(analysis, open(out_file, "wb"))
