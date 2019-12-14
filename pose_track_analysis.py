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
    return min(lelbow[1], relbow[1], wrist[1]) <= neck[1]


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
        #point = (basket[0] + basket_hspace - point[0], point[1] - (basket[1] - basket_vspace))

    if not (0 <= point[0] <= 1920) or point[1] <= 0:
        return None
    else:
        return point[0], point[1]


truth = {
    "42_dunk_386_521": {16: [441, 444]},
    "39_dunk_186_321": {5: [207]},
    "195_dunk_715_850": {},
    "167_dunk_672_807": {6: [738]},
    "164_dunk_237_372": {},
    "148_dunk_824_959": {},
    "82_dunk_2740_2875": {8: [2823]},
    "17_dunk_382_517": {},
    "130_dunk_479_614": {69: [579]},  # correct
    "2_three-pointer_97_232": {5: [159]},  # correct
    "10_dunk_405_540": {73: [459]},  # correct
    "5_three-pointer_844_979": {8: [885, 888]},  # correct
    "6_three-pointer_533_668": {4: [570, 573]},  #correct perfect
    "42_dunk_722_857": {12: [771, 774, 775], 8: [786]},  #correct
    "44_dunk_1660_1795": {15: [1713]},  # correct
    "169_dunk_888_1023": {},
    "71_dunk_1921_2056": {67: [1998]},  # semantically correct
    "38_dunk_1334_1469": {6: [1386]},  # noisy signal
    "28_dunk_1842_1977": {12: [1857]},  # early but ok signal
    "26_dunk_2464_2599": {11: [2559]},  # correct
    "155_dunk_1579_1714": {1: [1623]},  # early defender but ok signal
    "120_dunk_1529_1664": {9: [1593]},  # semantically correct
    "148_dunk_2435_2570": {21: [2508]},  # noisy but correct
    "147_dunk_1381_1516": {6: [1443]},  # noisy
    "132_dunk_1414_1549": {31: [1476]},  # arm may not be up
    "159_dunk_1725_1860": {},
    "6_three-pointer_1695_1830": {4: [1773]},  # 19 three pointers from here needed
    "165_three-pointer_1917_2052": {6: [1977]},  # correct perfect
    "14_three-pointer_2655_2790": {6: [2706]},  # correct perfect
    "13_three-pointer_4105_4240": {4: [4224]},  # incorrect but ok signal
    "158_three-pointer_1869_2004": {},
    "104_three-pointer_2688_2823": {7: [2748, 2751, 2754]},  # correct perfect
    "104_three-pointer_2794_2929": {35: [2880]},
    "104_three-pointer_3125_3260": {3: [3180, 3183]},  # good defender jump
    "13_three-pointer_3829_3964": {},
    "157_three-pointer_1255_1390": {3: [1329]},  # noisy
    "165_three-pointer_789_924": {6: [861, 864]},   # correct perfect
    "9_three-pointer_1339_1474": {43: [1422]},
    "104_three-pointer_992_1127": {39: [1041]},   # may result in 0 point
    "8_three-pointer_3014_3149": {15: [3048, 3051]},   # 8 more three pointers from here
    "165_three-pointer_2596_2731": {9: [2697]},   # a bit tricky defender jump. missing frame
    "165_three-pointer_2153_2288": {1: [2193], 9: [2193, 2196]},
    "166_three-pointer_2490_2625": {10: [2565, 2568, 2571]},
    "8_three-pointer_2456_2591": {9: [2541, 2544, 2547, 2550]},  # perfect defender
    "107_three-pointer_378_513": {4: [435, 438, 441, 444]},   # perfect stephen curry
    "164_three-pointer_1622_1757": {6: [1674, 1677, 1680, 1683]},  # correct perfect
    "159_three-pointer_1880_2015": {4: [1935, 1938, 1941, 1944]},  # correct perfect
    "13_three-pointer_3189_3324": {34: [3243, 3246, 3249, 3252]},  # correct perfect
    "158_three-pointer_2804_2939": {5: [2817, 2820, 2823, 2826, 2829]},  # arm was not up
    "158_three-pointer_2449_2584": {12: [2505]},
    "153_three-pointer_2186_2321": {24: [2253, 2256, 2259, 2262]},  # correct perfect
    "5_three-pointer_412_547": {2: [423]},  # noisy.. missing first couple of frames
    "8_three-pointer_318_453": {13: [405, 408]},
    "153_three-pointer_827_962": {9: [870, 873, 876]},
    "5_three-pointer_1212_1347": {6: [1254, 1257, 1260, 1263, 1266]},  # correct perfect
    "76_dunk_1473_1608": {14: [1539, 1542, 1545]},
    "74_dunk_2657_2792": {36: [2727, 2730, 2733]},  # correct perfect
    "167_dunk_4076_4211": {16: [4122, 4125, 4128, 4131, 4143]},  # noisy, but ok
    "183_dunk_1596_1731": {26: [2202]},
    "178_dunk_1944_2079": {},
    "142_dunk_1719_1854": {15: [1782, 1785]},
    "7_dunk_143_278": {8: [192, 195, 198]}
}


def process(tracks, unmatched_poses):
    player_travels = {}
    player_positions = {}
    shifted_positions = {}
    jumps = list()

    negative_jumps = {}
    for i in range(8):
        negative_jumps[i] = list()
    tmove = list()
    tjump = list()
    tmj = list()

    key_frames = []
    frame_ids = sorted(all_baskets.keys())
    frame_offset = frame_ids[0]
    no_truth = video_short_name not in truth
    for track in tracks:
        truth_frame = list()
        if video_short_name in truth and track in truth[video_short_name]:
            truth_frame = truth[video_short_name][track]

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
            frame_i = frame_ids[i]
            max_travel = np.max(player_travel[1][i - 3:i])  # must be less than -25
            min_travel = np.min(player_travel[1][i - 3:i])  # must be less than -50
            arms_count = player_travel[2][i]  # must be greater than or equal to 1
            position = shifted_position[:, i]
            x = position[0]
            y = position[1]
            if not left_basket:
                x = im_width - x

            has_jump = np.argwhere(player_travel[1][i - 3:i] < -25).shape[0] == 3 and min_travel < -50 \
                       and arms_count >= 1 and shifted_position[1][i] > 0

            if frame_i in truth_frame or (has_jump and no_truth):
                #print(f"Track: {track}, jumped around frame: {frame_i}")
                jump = {'end_frame_id': frame_ids[i], 'track_id': track, 'basket': get_basket(frame_i),
                        'player_position': player_position[:, i], 'shifted_position': position,
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

                # save as training data
                class_label = 1 if "dunk" in track_file else 2
                tmove.append((track, frame_i, class_label, f'1:{x}', f'2:{y}'))
                tjump.append((track, frame_i, 1, f'1:{max_travel}', f'2:{min_travel}', f'3:{arms_count}'))
                tmj.append((track, frame_i, class_label,
                            f'1:{x}', f'2:{y}', f'3:{max_travel}', f'4:{min_travel}', f'5:{arms_count}'))
            else:
                # Generate negative training examples
                if max_travel < -25 and min_travel < -50 and arms_count == 0:
                    negative_jumps[0].append((track, frame_ids[i], max_travel, min_travel, arms_count, x, y))

                elif max_travel < -25 and min_travel > -50 and arms_count >= 1:
                    negative_jumps[1].append((track, frame_ids[i], max_travel, min_travel, arms_count, x, y))
                elif max_travel < -25 and 0 > min_travel > -50 and arms_count >= 1:
                    negative_jumps[7].append((track, frame_ids[i], max_travel, min_travel, arms_count, x, y))

                elif max_travel > -25 and min_travel < -50 and arms_count >= 1:
                    negative_jumps[2].append((track, frame_ids[i], max_travel, min_travel, arms_count, x, y))
                elif 0 > max_travel > -25 and min_travel < -50 and arms_count >= 1:
                    negative_jumps[8].append((track, frame_ids[i], max_travel, min_travel, arms_count, x, y))

                elif max_travel > -25 and min_travel > -50 and arms_count >= 1:
                    negative_jumps[3].append((track, frame_ids[i], max_travel, min_travel, arms_count, x, y))
                elif max_travel > -25 and min_travel < -50 and arms_count == 0:
                    negative_jumps[4].append((track, frame_ids[i], max_travel, min_travel, arms_count, x, y))
                elif max_travel < -25 and min_travel > -50 and arms_count == 0:
                    negative_jumps[5].append((track, frame_ids[i], max_travel, min_travel, arms_count, x, y))
                elif max_travel > -25 and min_travel > -50 and arms_count == 0:
                    negative_jumps[6].append((track, frame_ids[i], max_travel, min_travel, arms_count, x, y))

    for values in negative_jumps.values():
        if len(values) > 0:
            v = values[np.random.randint(0, len(values))]
            tjump.append((v[0], v[1], -1, f'1:{v[2]}', f'2:{v[3]}', f'3:{v[4]}'))
            tmj.append((v[0], v[1], -1, f'1:{v[5]}', f'2:{v[6]}', f'3:{v[2]}', f'4:{v[3]}', f'5:{v[4]}'))

    return jumps, player_travels, player_positions, shifted_positions, key_frames, tmove, tjump, tmj


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
    for frame in frame_ids[1:]:
        step = int((frame - previous_frame_id) / basket_frame_ratio)
        for i in range(step-1):
            missing_frames[previous_frame_id + basket_frame_ratio * (i + 1)] = \
                fit_missing_point(baskets[frame][0], baskets[previous_frame_id][0], step-1, i+1), \
                fit_missing_point(baskets[frame][1], baskets[previous_frame_id][1], step-1, i+1)
        previous_frame_id = frame

    baskets.update(missing_frames)

    left_count = 0  # How many times did we see the basket on left half of the frame
    right_count = 0
    for basket in baskets.values():
        basket_x = basket[0][0] + basket[1][0]
        if basket_x < 0.9*im_width:
            left_count += 1
        elif basket_x > 1.1*im_width:
            right_count += 1

    return left_count > right_count


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


def print_training_file(dataset, filename_suffix):
    if len(dataset) == 0:
        return

    file = open(out_train_prefix + "_" + filename_suffix, 'w')
    for row in dataset:
        s = video_name
        for x in row:
            s += f' {x}'
        print(s, file=file)


if __name__ == '__main__':
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_track_pkl', type=str, required=True, help='pose track pkl file name')
    parser.add_argument('--basket_pkl', type=str, required=True, help='basket bbox pkl file name')

    args = parser.parse_args()
    track_file = args.pose_track_pkl
    out_jump_file = track_file.rsplit(".", 1)[0] + "_jumps.pkl"
    out_keyframes_file = track_file.rsplit(".", 1)[0] + "_keyframes.txt"
    out_train_prefix = track_file.rsplit("_", 2)[-3]
    video_name = out_train_prefix.rsplit("/", 1)[-1]
    video_short_name = video_name.rsplit(".", 1)[-1]

    print(f"Processing {video_name}")
    basket_frame_ratio = 3
    im_width = 1920
    im_height = 1080
    basket_width = 70
    basket_height = 50
    basket_vspace = 50
    basket_hspace = 190

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

    jumps, player_travels, player_positions, shifted_positions, key_frames, tmove, tjump, tmj = process(tracks, poses)

    pickle.dump((jumps, player_travels, player_positions, shifted_positions), open(out_jump_file, "wb"))
    print(key_frames, file=open(out_keyframes_file, 'w'))

    #if not basket_location_unknown:
    print_training_file(tmove, "move_labels.txt")
    print_training_file(tjump, "jump_labels.txt")
    print_training_file(tmj, "mj_labels.txt")

    for key in truth.keys():
        print(key)
