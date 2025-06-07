import math
import numpy as np


def load_garage_label(dir, state="train"):
    with open(dir + "/garage/with_shade/garage_orient.txt") as f:
        garage_orient = f.readlines()
    garage_info_list = []
    train_num_rate = 0.8
    for i, line in enumerate(garage_orient):
        infos = [float(info) for info in line.strip().split(" ")]
        if len(infos) == 5:
            garage_info = {}
            garage_info["file_name"] = i + 1
            garage_info["center"] = [infos[0] + infos[2] / 2, infos[1] + infos[3] / 2]
            garage_info["orient"] = 180 - np.rad2deg(math.atan2(infos[3], infos[2])) if infos[4] == -1 else -np.rad2deg(
                math.atan2(infos[3], infos[2]))
            garage_info_list.append(garage_info)
        else:
            assert len(infos) == 10
            garage_info = {}
            garage_info["file_name"] = i + 1
            garage_info["center"] = [infos[0] + infos[2] / 2, infos[1] + infos[3] / 2]
            garage_info["orient"] = 180 - np.rad2deg(math.atan2(infos[3], infos[2])) if infos[4] == -1 else -np.rad2deg(
                math.atan2(infos[3], infos[2]))
            garage_info_list.append(garage_info)
            garage_info = {}
            garage_info["file_name"] = i + 1
            garage_info["center"] = [infos[5] + infos[7] / 2, infos[6] + infos[8] / 2]
            garage_info["orient"] = 180 - np.rad2deg(math.atan2(infos[8], infos[7])) if infos[9] == -1 else -np.rad2deg(
                math.atan2(infos[8], infos[7]))
            garage_info_list.append(garage_info)
    if state == "train":
        return garage_info_list[:int(len(garage_info_list) * train_num_rate)]
    if state == "test":
        return garage_info_list[int(len(garage_info_list) * train_num_rate):]


def load_plane_label(dir, state="train"):
    with open(dir + "/planes/labelTXT.txt") as f:
        plane_label = f.readlines()
    plane_info_list = []
    train_num_rate = 0.8
    for i, line in enumerate(plane_label[1:]):
        infos = line.strip().split(" ")
        plane_info = {}
        plane_info["file_name"] = i + 1
        plane_info["angle"] = 90 - float(infos[0])
        plane_info_list.append(plane_info)
    if state == "train":
        return plane_info_list[:int(len(plane_info_list) * train_num_rate)]
    if state == "test":
        return plane_info_list[int(len(plane_info_list) * train_num_rate):]


def load_garage_location(dir, state):
    with open(dir + "/garage_deposit_location.txt") as f:
        garage_location = f.readlines()
    garage_location_info_list = []
    train_num_rate = 0.8
    for i, line in enumerate(garage_location):
        infos = line.strip().split(", ")
        garage_location_info = {}
        garage_location_info["index"] = i
        garage_location_info["location"] = [int(infos[0]), int(infos[1])]
        garage_location_info["orient"] = float(infos[2])
        garage_location_info_list.append(garage_location_info)
    if state == "train":
        return garage_location_info_list[:int(len(garage_location_info_list) * train_num_rate)]
    if state == "test":
        return garage_location_info_list[int(len(garage_location_info_list) * train_num_rate):]