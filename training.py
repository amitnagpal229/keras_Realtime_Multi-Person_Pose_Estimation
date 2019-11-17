import os
import sys
import argparse
import pickle
import json

dir_path = "/Volumes/GoogleDrive/My Drive/CS229/project/trainingData/"


def add_entry(segments, plays, video, key):
    for entry in plays:
        frame_index = entry["frame_index"]
        index = key+'-'+str(frame_index)
        if "dunk" in entry["description"]:
            segments[index] = ("dunk", entry["frame_index"], video)
        elif "three-pointer" in entry["description"]:
            segments[index] = ("three-pointer", entry["frame_index"], video)


if __name__ == '__main__':
    files = os.listdir(dir_path)
    segments = {}

    for entry in files:
        print(entry)
        if "output" in entry:
            fp = open(dir_path + entry, "r")
            fp_in = open(dir_path + entry.replace("output", "input"))
            video_url = json.loads(fp_in.readline())["videoUrl"]
            if "turner" in video_url:
                plays = []
                for line in fp:
                    if len(line) > 0:
                        dict = json.loads(line)
                        if "plays" in dict:
                            add_entry(segments, dict["plays"], video_url, entry)
                        if "otherPlays" in dict:
                            add_entry(segments, dict["otherPlays"], video_url, entry)

    print(segments)
    print(len(segments))
    video_list = {}
    for key, value in segments.items():
        print(f"{key.rsplit('-', 1)[0]} {value[2]}")
        video_list[key.rsplit("-", 1)[0]] = value[2]
