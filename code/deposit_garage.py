import cv2
import numpy as np
import math
from combing_plane_garage import *
from load_txt import *


def consistent_images(foreground_img, background, location_garage_rotate, loacation_garage_airport, return_background=False):
    """取两者交集"""
    fore_h, fore_w = foreground_img.shape[:2]
    back_h, back_w = background.shape[:2]
    h_start_back = max(0, loacation_garage_airport[1] - location_garage_rotate[1])
    w_start_back = max(0, loacation_garage_airport[0] - location_garage_rotate[0])
    h_end_back = min(back_h, fore_h + loacation_garage_airport[1] - location_garage_rotate[1])
    w_end_back = min(back_w, fore_w + loacation_garage_airport[0] - location_garage_rotate[0])
    background_crop_img = background[h_start_back: h_end_back, w_start_back: w_end_back]

    h_start_fore = max(0,  location_garage_rotate[1] - loacation_garage_airport[1])
    w_start_fore = max(0,  location_garage_rotate[0] - loacation_garage_airport[0])
    h_end_fore = min(fore_h, back_h + location_garage_rotate[1] - loacation_garage_airport[1])
    w_end_fore = min(fore_w, back_w + location_garage_rotate[0] - loacation_garage_airport[0])
    foreground_crop_img = foreground_img[h_start_fore: h_end_fore, w_start_fore: w_end_fore]
    assert background_crop_img.shape == foreground_crop_img.shape
    if return_background == True:
        located_img = overlay_images(background_crop_img, foreground_crop_img)
    else:
        located_img = foreground_crop_img
    return located_img, [h_start_back, h_end_back, w_start_back, w_end_back], \
           [loacation_garage_airport[0] - w_start_back, loacation_garage_airport[1] - h_start_back]
           # [location_garage_rotate[1] - h_start_fore, location_garage_rotate[0] - w_start_fore]



def deposit_background(foreground_img, background, location_garage, garage_info, loacation_garage_airport, return_background=True):
    convert_angle = loacation_garage_airport["orient"] - garage_info["orient"]
    foreground_img, point_rotate = rotate_image(foreground_img, convert_angle, [[location_garage[0], location_garage[1]]])
    location_garage_rotate = [int(point_rotate[0][0]), int(point_rotate[0][1])]
    # location_garage_rotate = location_garage
    return consistent_images(foreground_img, background, location_garage_rotate, loacation_garage_airport["location"], return_background)


import random
if __name__ == "__main__":

    background_img = cv2.imread('processed_data/background_1/background.png', cv2.IMREAD_UNCHANGED)
    dir = "processed_data/background_1"
    garage_info_list = load_garage_label(dir)
    plane_info_list = load_plane_label(dir)
    garage_location_list = load_garage_location(dir)


    index_back = random.randint(0, 20)
    index_plane = random.randint(0, 6)
    index_garage = random.randint(0, 8)

    combing_img, location_garage, plane_size = combine_plane_garage('processed_data/background_1', index_plane, index_garage, plane_info_list, garage_info_list, need_plane_size=True)
    located_img, [h_start, h_end, w_start, w_end], center_relative = deposit_background(combing_img, background_img, location_garage, garage_info_list[index_garage], garage_location_list[index_back])
    cv2.imwrite('locate_img.png', located_img)

    with open(dir + "/data_for_train/ground_truth.txt", 'w') as file:
        file.write(f"{index_plane, index_garage, index_back, garage_location_list[index_back]['location'][0], garage_location_list[index_back]['location'][1], plane_size[1], plane_size[0], garage_location_list[index_back]['orient']}\n")





