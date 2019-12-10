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
        return value - low if value - low > -200 else 0
    else:
        return value - high if value - low > -200 else 0


def shift_point(point, frame):
    if point is None:
        return None
    elif basket_location_unknown:
        return point

    basket = np.array(get_basket(frame))
    if left_basket:
        point = np.array(point) - basket + np.array((basket_hspace, basket_vspace))
    else:
        point = np.array(point) - basket + np.array((im_width-basket_hspace, basket_vspace))

    if not (0 <= point[0] <= 1920) or point[1] <= 0:
        return None
    else:
        return point[0], point[1]


def process(tracks, unmatched_poses):
    player_travels = {}
    player_positions = {}
    shifted_positions = {}
    jumps = list()
    key_frames = []
    frame_ids = sorted(all_baskets.keys())
    frame_offset = frame_ids[0]
    for track in tracks:
        player_position = np.zeros((3, len(frame_ids)), dtype=np.int16)
        shifted_position = np.zeros((3, len(frame_ids)), dtype=np.int16)
        player_positions[track] = player_position
        shifted_positions[track] = shifted_position
        for frame in frame_ids:
            if frame in tracks[track]:
                pose = tracks[track][frame]
                hip = get_hip_joint(pose)
                hip_shifted = shift_point(hip, frame)
                if hip_shifted is not None:
                    frame_index = int((frame - frame_offset) / 3)
                    player_position[:, frame_index] = hip[0], hip[1], arms_up(pose)
                    shifted_position[:, frame_index] = hip_shifted[0], hip_shifted[1], arms_up(pose)

        player_travel = np.zeros((3, len(frame_ids)), dtype=np.int16)
        for i in range(1, len(frame_ids)):
            history = shifted_position[:, max(0, i - 5): i]
            player_travel[2][i] = np.sum(history[2]) + shifted_position[2][i]
            player_travel[1][i] = get_max_travel(history[1], shifted_position[1][i])
            player_travel[0][i] = get_max_travel(history[0], shifted_position[0][i])
        player_travels[track] = player_travel

        previous_jump_frame = -1
        for i in range(3, len(frame_ids)):
            if shifted_position[1][i] > 0 and np.argwhere(player_travel[1][i-3:i] < -30).shape[0] == 3 and np.min(player_travel[1][i-3:i]) < -55 and player_travel[2][i] >= 1:
                frame_i = frame_ids[i]
                print(f"Track: {track}, jumped around frame: {frame_i}")
                jump = {'end_frame_id': frame_ids[i], 'track_id': track, 'basket': get_basket(frame_i),
                        'player_position': player_position[:, i], 'shifted_position': shifted_position[:, i],
                        'player_travel': player_travel[:, i]}
                if frame_i in tracks[track]:
                    jump['pose'] = tracks[track][frame_i]
                jump['previous_frame_ids'] = list()
                for j in range(i-3, i):
                    frame_j = frame_ids[j]
                    if frame_j in tracks[track]:
                        jump['previous_frame_ids'].append({'frame_id': frame_j,
                                                           'basket': get_basket(frame_j),
                                                           'pose': tracks[track][frame_j],
                                                           'player_position': player_position[:, j],
                                                           'shifted_position': shifted_position[:, j],
                                                           'player_travel': player_travel[:, j]
                                                           })
                if previous_jump_frame == i-1:
                    del key_frames[-1]
                    del jumps[-1]
                jumps.append(jump)
                key_frames.append(jump['end_frame_id'])
                previous_jump_frame = i

    return jumps, player_travels, player_positions, shifted_positions, key_frames


def get_basket(frame_id):
    if frame_id in all_baskets:
        xy = all_baskets[frame_id][0]
        return xy[0]+int(basket_width/2), xy[1]-int(basket_height/2)
    elif left_basket:
        return 0, 0
    else:
        return im_width-1, 0


# Fills baskets in frames that don't have a basket
# Returns True if basket is on the left, else false if it is on the right
def smooth_baskets(baskets):
    frame_ids = sorted(baskets.keys())
    missing_frames = {}
    previous_frame_id = frame_ids[0]
    left_count = 0  # How many times did we see the basket on left half of the frame
    for frame in frame_ids[1:]:
        if baskets[frame][0][0] < im_width / 2:
            left_count += 1
        step = int((frame - previous_frame_id) / basket_frame_ratio)
        for i in range(step-1):
            missing_frames[previous_frame_id + basket_frame_ratio * (i + 1)] = \
                fit_missing_point(baskets[frame][0], baskets[previous_frame_id][0], step-1, i+1), \
                fit_missing_point(baskets[frame][1], baskets[previous_frame_id][1], step-1, i+1)
        previous_frame_id = frame

    baskets.update(missing_frames)
    return left_count > len(frame_ids) / 2


def fit_missing_point(left, right, num_missing, position):
    l = np.array(left)
    r = np.array(right)
    missing = l + position * (r - l) / (num_missing + 1)
    return int(missing[0]), int(missing[1])


def get_court_xy_limits(basket):
    x = 0
    y = (max(0, basket[1] - basket_vspace), im_height)
    if left_basket:
        x = (max(0, basket[0] - basket_hspace), im_width)
    else:
        x = (0, min(im_width, basket[0] + args.basket_hspace))
    return x, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_track_pkl', type=str, required=True, help='pose track pkl file name')
    parser.add_argument('--basket_pkl', type=str, required=True, help='basket bbox pkl file name')

    args = parser.parse_args()
    track_file = args.pose_track_pkl
    out_jump_file = track_file.rsplit(".", 1)[0] + "_jumps.pkl"
    out_keyframes_file = track_file.rsplit(".", 1)[0] + "_keyframes.txt"

    basket_frame_ratio = 3
    im_width = 1920
    im_height = 1080
    basket_width = 70
    basket_height = 50
    basket_vspace = 50
    basket_hspace = 140

    tracks, poses = pickle.load(open(track_file, "rb"))
    all_baskets = {}
    basket_location_unknown = False
    try:
        all_baskets = pickle.load(open(args.basket_pkl, "rb"))
        left_basket = smooth_baskets(all_baskets)
    except:
        for frame in poses:
            all_baskets[frame] = ((0, 0), (0, 0))
        left_basket = True
        basket_location_unknown = True
        print("Basket file not found, will not adjust for camera motion")

    jumps, player_travels, player_positions, shifted_positions, key_frames = process(tracks, poses)

    pickle.dump((jumps, player_travels, player_positions, shifted_positions), open(out_jump_file, "wb"))
    print(key_frames, file=open(out_keyframes_file, 'w'))
