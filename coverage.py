import os
import sys
import argparse
import cv2
import time
import pickle
import numpy as np
import math
import json
from config_reader import config_reader
import util

bp_id_to_str = ["nose", "neck", "rshoulder", "relbow", "rwrist", "lshoulder", "lelbow", "lwrist", "rhip", "rknee",
                "rankle", "lhip", "lknee", "lankle", "reye", "leye", "rear", "lear"]
bp_str_to_id = {}
for i, part in enumerate(bp_id_to_str):
    bp_str_to_id[part] = i

limb_id_to_str = ["rshoulder", "lshoulder", "rupperarm", "rlowerarm", "lupperarm", "llowerarm", "rhipneck",
                  "rupperleg", "rlowerleg", "lhipneck", "lupperleg", "llowerleg", "noseneck", "reyenose", "reyeear",
                  "leyenose", "leyeear"]
limb_str_to_id = {}
for i, limb in enumerate(limb_id_to_str):
    limb_str_to_id[limb] = i


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', type=str, required=True, help='directory containing features pkl files')
    parser.add_argument('--generate_features', type=str, help='directory containing features pkl files')

    args = parser.parse_args()
    file_dir = args.file_dir

    persons = 0
    frames = 0
    parts = np.zeros((len(bp_id_to_str),))
    limbs = np.zeros((len(limb_id_to_str),))
    files = os.listdir(file_dir)
    count = 0
    for file in files:
        count += 1
        print(f"{count}) {file}")
        features = pickle.load(open(file_dir + file, "rb"))
        for frame in features:
            frames += 1
            body_parts = features[frame]["body_parts"]
            people = features[frame]["people"]
            for part in body_parts:
                parts[bp_str_to_id[part]] += len(body_parts[part])
            for person in people:
                persons += 1
                for limb in person:
                    limbs[limb_str_to_id[limb]] += 1

    def getCounts(plist):
        count = 0
        for p in plist:
            count += parts[bp_str_to_id[p]]
        return count / frames, count / persons

    def getLimbCounts(plist):
        count = 0
        for p in plist:
            count += limbs[limb_str_to_id[p]]
        return count / frames, count / persons

    print(f"persons={persons}, frames={frames}, persons/frame={persons/frames}")
    print(f'knee: {getCounts(("lknee", "rknee"))}')
    print(f'wrist: {getCounts(("lwrist", "rwrist"))}')
    print(f'ankle: {getCounts(("lankle", "rankle"))}')
    print(f'elbow: {getCounts(("lelbow", "relbow"))}')
    print(f'neck: {getCounts(["neck"])}')
    print(f'shoulder: {getLimbCounts(("lshoulder", "rshoulder"))}')
    print(f'upper-leg: {getLimbCounts(("lupperleg", "rupperleg"))}')
    print(f'upper-arm: {getLimbCounts(("lupperarm", "rupperarm"))}')
    print(f'lower-leg: {getLimbCounts(("llowerleg", "rlowerleg"))}')
    print(f'lower-arm: {getLimbCounts(("llowerarm", "rlowerarm"))}')

    if args.generate_features is not None:
        pose = pickle.load(open(args.generate_features, "rb"))

        video_positions = {}
        for frame in pose:
            all_peaks, subset, candidate = pose[frame]

            body_parts = {}  # body part id -> list of all instances (across all people)
            for i in range(len(bp_id_to_str)):
                key = bp_id_to_str[i]
                body_parts[key] = list()
                for peak in all_peaks[i]:
                    body_parts[key].append({"x": int(peak[0]), "y": int(peak[1]), "s": int(peak[2])})

            people = list()  # list of persons. person is a list of LimbLocations
            for person_id in range(len(subset)):
                person = {}
                for limb in range(len(limb_id_to_str)):
                    key = limb_id_to_str[limb]
                    index = subset[person_id][np.array(util.limbSeq[limb])-1]
                    if -1 in index:
                        continue
                    X = candidate[index.astype(int), 0]
                    Y = candidate[index.astype(int), 1]
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                    person[key] = {"from": (int(X[0]), int(Y[0])), "to": (int(X[1]), int(Y[1])), "len": float(length), "ang": float(angle), "s": 1}
                people.append(person)

            video_positions[frame] = {"body_parts": body_parts, "people": people}

        feature_file = args.generate_features.rsplit("_", 1)[0] + "_features"
        pickle.dump(video_positions, open(feature_file + ".pkl", "wb"))
        json.dump(video_positions, open(feature_file + ".json", "w"), sort_keys=True, indent=4, separators=(',', ': '))
