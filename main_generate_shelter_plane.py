import cv2
import numpy as np
import math
from code_shelter_plane.combing_plane_garage import *
from code_shelter_plane.load_txt import *
from code_shelter_plane.deposit_garage import *


def move_bbox_to_half(x, y, w, h, angle):
    # 将中心点移动到原中心点在角度方向上的一半位置
    rad = np.deg2rad(angle)
    new_cx = x + h / 4 * math.cos(rad)
    new_cy = y + h / 4 * math.sin(rad)
    return new_cx, new_cy, w, h, angle


def load_garage(dir, garage_info):
    garage_foreground = cv2.imread(dir + "/garage/with_shade/{}_foreground.png".format(garage_info["file_name"]),
                                   cv2.IMREAD_UNCHANGED)
    if os.path.exists(dir + "/garage/with_shade/{}_shade.png".format(garage_info["file_name"])):
        garage_shade = cv2.imread(dir + "/garage/with_shade/{}_shade.png".format(garage_info["file_name"]),
                                  cv2.IMREAD_UNCHANGED)
    else:
        garage_shade = np.zeros_like(garage_foreground)
    garage_img = overlay_images(garage_shade, garage_foreground)
    return garage_img


if __name__ == "__main__":
    img_w = 0
    img_h = 0
    dir = 'processed_data/background_1'
    background_img = cv2.imread(dir + '/background.png', cv2.IMREAD_UNCHANGED)
    # 获取图像的尺寸
    height, width, _ = background_img.shape
    block_size = 1024
    state = "train"
    garage_info_list = load_garage_label(dir, state)
    plane_info_list = load_plane_label(dir, state)
    garage_location_list = load_garage_location(dir, state)

    imgs_path = []
    label_info = []
    # for h in range(0, height, block_size):
    #     for w in range(0, width, block_size):
    if state == "train":
        dir_save_img = dir + "/data_for_train/train/img"
        dir_save_label = dir + "/data_for_train/train/label"
    else:
        dir_save_img = dir + "/data_for_train/test/img"
        dir_save_label = dir + "/data_for_train/test/label"

    if not os.path.exists(dir_save_label):
        os.makedirs(dir_save_label)
    if not os.path.exists(dir_save_img):
        os.makedirs(dir_save_img)


    for wzl in range(600):
        h, w = random.randint(0, height - block_size), random.randint(0, width - block_size)
        h_start, h_end, w_start, w_end = h, h + block_size, w, w + block_size
        if h + block_size > height:
            h_start, h_end = h_start - (h + block_size - height), h_end - (h + block_size - height)
        if w + block_size > width:
            w_start, w_end = w_start - (w + block_size - width), w_end - (w + block_size - width)
        block_filename = dir_save_img + "/block_{}_{}.jpg".format(h_start, w_start)
        if block_filename in imgs_path:
            continue

        background_img_part = background_img[h_start: h_end, w_start: w_end]
        garage_location_part_list = [location for location in garage_location_list if
                                     h_start < location["location"][1] < h_end and
                                     w_start < location["location"][0] < w_end]
        garage_location_part_relative_list = [[location["location"][0] - w_start, location["location"][1] - h_start]
                                              for location in garage_location_part_list]

        max_num = len(garage_location_part_list)
        if max_num == 0:
            continue
        random_index_garage_location = random.sample(range(0, max_num), int(max_num * 0.7)) if max_num > 1 else [0]
        for index_garage_location in random_index_garage_location:
            h_star = max(h, 0)
            w_star = max(w, 0)
            h_end = min(h + block_size, height)
            w_end = min(w + block_size, width)

            index_plane = random.randint(0, len(plane_info_list) - 1)
            index_garage = random.randint(0, len(garage_info_list) - 1)
            index_back = index_garage_location
            # index_garage = 1
            # index_plane = 4
            # index_back = 0
            # print(index_garage, index_plane, index_back)
            if random.random() < 0.4: # 只有机库，无飞机
                plane_inside = False
                combing_img, location_garage = load_garage(dir, garage_info_list[index_garage]), garage_info_list[index_garage]["center"]
            else:
                plane_inside = True
                combing_img, location_garage, plane_size = combine_plane_garage(dir, plane_info_list[index_plane],
                                                                                garage_info_list[index_garage],
                                                                                need_plane_size=True)
            located_img_on_background, location_range_on_total, center_relative = deposit_background(combing_img, background_img,
                                                                      location_garage,
                                                                      garage_info_list[index_garage],
                                                                      garage_location_part_list[index_back],
                                                                      return_background=False)
            # cv2.imwrite('locate_img.png', located_img_on_background)
            located_img_part_img, location_range_on_part, _ = consistent_images(located_img_on_background, background_img_part, center_relative, garage_location_part_relative_list[index_back])
            empty_img_part = np.zeros_like(background_img_part)
            empty_img_part[location_range_on_part[0]: location_range_on_part[1], location_range_on_part[2]: location_range_on_part[3]] = located_img_part_img
            background_img_part = overlay_images(background_img_part, empty_img_part)

            if plane_inside == True:
                x_, y_, w_, h_, angle_ = garage_location_part_relative_list[index_back][0], garage_location_part_relative_list[index_back][1], plane_size[1], plane_size[0], garage_location_part_list[index_back]['orient']
                angle_ = -angle_ - 90
                angle_ = angle_ % 360
                angle_ = angle_ - 360 if angle_ > 180 else angle_
                x_result, y_result, w_result, h_result, angle_result = move_bbox_to_half(x_, y_, w_, h_, angle_)
                label_info.append(
                    [index_plane, index_garage, index_back, x_result, y_result, w_result, h_result, angle_result])

        block_filename_save = os.path.join(block_filename)
        cv2.imwrite(block_filename_save, background_img_part)
        with open(dir_save_label + "/{}.txt".format(block_filename[52:-4]), 'w') as file:
            for info in label_info:
                line = ' '.join(map(str, info))
                file.write(f"{line}\n")
        label_info = []
        imgs_path.append(block_filename)

    # for file_name in imgs_path:
    #     with open(dir + "/data_for_train/labelTXT/{}.txt".format(file_name[47:]), 'w') as file:
    #         for info in label_info:
    #             line = ' '.join(map(str, info))
    #             file.write(f"{line}\n")
    # with open(dir + "/data_for_train/imgs_path.txt", 'w') as file:
    #     for info in imgs_path:
    #         file.write(f"{info}\n")