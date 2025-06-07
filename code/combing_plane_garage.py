import os
import cv2
import numpy as np
import cv2
import numpy as np
import math
from combing_plane_garage import *
from load_txt import *
from deposit_garage import *

def combine_plane_garage(dir, plane_info, garage_info, need_plane_size = False):
    garage_foreground = cv2.imread(dir + "/garage/with_shade/{}_foreground.png".format(garage_info["file_name"]), cv2.IMREAD_UNCHANGED)
    if os.path.exists(dir + "/garage/with_shade/{}_shade.png".format(garage_info["file_name"])):
        garage_shade = cv2.imread(dir + "/garage/with_shade/{}_shade.png".format(garage_info["file_name"]), cv2.IMREAD_UNCHANGED)
    else:
        garage_shade = np.zeros_like(garage_foreground)
    plane_shade = cv2.imread(dir + "/planes/{}.png".format(plane_info["file_name"]), cv2.IMREAD_UNCHANGED)
    if os.path.exists(dir + "/planes/{}_no_shade.png".format(plane_info["file_name"])):
        plane_single = cv2.imread(dir + "/planes/{}_no_shade.png".format(plane_info["file_name"]), cv2.IMREAD_UNCHANGED)
    else:
        plane_single = plane_shade
    shade_range_image = cv2.imread(dir + "/shade_color.png", cv2.IMREAD_UNCHANGED)

    assert plane_shade.shape == plane_single.shape and garage_foreground.shape == garage_shade.shape

    # 旋转飞机为正方向
    plane_single_rotate = rotate_image(plane_single, plane_info["angle"])
    plane_shade_rotate = rotate_image(plane_shade, plane_info["angle"])
    plane_shade_rotate, mark, _ = crop_image(plane_shade_rotate)
    plane_single_rotate = plane_single_rotate[mark[0]: mark[1]+1, mark[2]: mark[3]+1]

    # 飞机尺寸变化
    scale_percent = 1
    width = int(plane_single.shape[1] * scale_percent)
    height = int(plane_single.shape[0] * scale_percent)
    dim = (width, height)
    plane_single_adjust = cv2.resize(plane_single_rotate, dim, interpolation=cv2.INTER_AREA)
    plane_shade_adjust = cv2.resize(plane_shade_rotate, dim, interpolation=cv2.INTER_AREA)

    # 飞机进行旋转后位移
    _, mark_single_plane, plane_size = crop_image(plane_single_adjust)
    location_plane_relate = [mark_single_plane[2] + plane_size[1] / 2,
                             mark_single_plane[0] + plane_size[0] / 2]
    plane_single_adjust, location_plane_relate = rotate_image(plane_single_adjust, garage_info["orient"],
                                                              [location_plane_relate])
    plane_shade_adjust = rotate_image(plane_shade_adjust, garage_info["orient"])

    _, _, padding_value_garage, padding_value_plane = padding_image(garage_foreground, plane_single_adjust)
    location_plane = [location_plane_relate[0][0] + padding_value_plane[2],
                      location_plane_relate[0][1] + padding_value_plane[0]]
    location_garage = [garage_info["center"][0] + padding_value_garage[2],
                       garage_info["center"][1] + padding_value_garage[0]]
    shift_distance = [int(location_garage[0] - location_plane[0]), int(location_garage[1] - location_plane[1])]
    plane_single_adjust = shift_image(plane_single_adjust, shift_distance[0], shift_distance[1])
    plane_shade_adjust = shift_image(plane_shade_adjust, shift_distance[0], shift_distance[1])

    # 将飞机与机库大小调为一致并裁剪掉机库内飞机部分
    assert garage_shade.shape == garage_foreground.shape
    garage_foreground, plane_single_adjust, padding_value_garage_1, padding_value_plane_1 = padding_image(garage_foreground, plane_single_adjust)
    garage_shade, plane_shade_adjust, _, _ = padding_image(garage_shade, plane_shade_adjust)
    location_garage_1 = [garage_info["center"][0] + padding_value_garage_1[2],
                         garage_info["center"][1] + padding_value_garage_1[0]]
    garage_mask = create_mask_with_line(plane_shade_adjust.shape, location_garage_1, garage_info["orient"])
    plane_single_adjust[garage_mask != 1, 3] = 0
    plane_shade_adjust[garage_mask != 1, 3] = 0

    # 分离阴影中飞机的部分和阴影外部分
    shade_mask_in = get_mask(garage_shade, 150)
    shade_mask_out = get_mask(garage_shade, 250)
    masked_plane_single = np.zeros_like(plane_single_adjust)
    masked_plane_single[shade_mask_in == 1] = plane_single_adjust[shade_mask_in == 1]
    masked_plane_out = np.zeros_like(plane_shade_adjust)
    # 将边缘处进行虚化（下一步）
    masked_plane_out[shade_mask_out == 0] = plane_shade_adjust[shade_mask_out == 0]

    # 构造阴影中飞机阴影部分
    masked_shade = np.zeros_like(plane_shade_adjust)
    height, width, _ = masked_shade.shape
    height_shade, width_shade, _ = shade_range_image.shape
    random_indices_y = np.random.randint(0, height_shade, size=(height, width))
    random_indices_x = np.random.randint(0, width_shade, size=(height, width))
    masked_shade[:, :, :3] = shade_range_image[random_indices_y, random_indices_x, :3]
    masked_shade[masked_plane_single[:, :, 3] != 0, 3] = np.minimum(masked_plane_single[masked_plane_single[:, :, 3] != 0, 3], garage_shade[masked_plane_single[:, :, 3] != 0, 3]) * 0.6

    # 叠加图层
    masked_plane_single = overlay_images(masked_plane_single, masked_shade)
    plane_garage = overlay_images(masked_plane_single, garage_foreground)
    planeshade_garage = overlay_images(masked_plane_out, plane_garage)
    planeshade_garageshade = overlay_images(garage_shade, planeshade_garage)
    if need_plane_size == True:
        return planeshade_garageshade, [int(location) for location in location_garage_1], plane_size
    return planeshade_garageshade, [int(location) for location in location_garage_1]


def overlay_images(overlay_1, overlay_2):
    """叠加两张图像，overlay_2是在上层的"""
    # 分离图像通道
    b_ol_1, g_ol_1, r_ol_1, a_ol_1 = cv2.split(overlay_1)
    b_ol_2, g_ol_2, r_ol_2, a_ol_2 = cv2.split(overlay_2)
    # 归一化 alpha 通道
    alpha_ol_1 = a_ol_1 / 255.0
    alpha_ol_2 = a_ol_2 / 255.0
    # 计算叠加后的图像
    alpha = (alpha_ol_1 + alpha_ol_2 - alpha_ol_1 * alpha_ol_2)
    b = (b_ol_1 * alpha_ol_1 * (1 - alpha_ol_2) + b_ol_2 * alpha_ol_2) / (alpha + 1e-10)
    g = (g_ol_1 * alpha_ol_1 * (1 - alpha_ol_2) + g_ol_2 * alpha_ol_2) / (alpha + 1e-10)
    r = (r_ol_1 * alpha_ol_1 * (1 - alpha_ol_2) + r_ol_2 * alpha_ol_2) / (alpha + 1e-10)
    alpha = 255 * alpha

    # 转换为8位图像并重新组合通道
    b = np.clip(b, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    r = np.clip(r, 0, 255).astype(np.uint8)
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)

    result = cv2.merge((b, g, r, alpha))
    return result


def padding_image(foreground, background, point = None):
    """
    两个变成两者中较大的
    :param foreground:
    :param background:
    :return:
    """
    target_height, target_width = max(foreground.shape[0], background.shape[0]), max(foreground.shape[1], background.shape[1])

    current_height, current_width = foreground.shape[:2]
    top_pad_1 = (target_height - current_height) // 2
    bottom_pad_1 = target_height - current_height - top_pad_1
    left_pad_1 = (target_width - current_width) // 2
    right_pad_1 = target_width - current_width - left_pad_1
    foreground_padding = cv2.copyMakeBorder(foreground, top_pad_1, bottom_pad_1, left_pad_1, right_pad_1, cv2.BORDER_CONSTANT,
                                            value=(0, 0, 0, 0))

    current_height, current_width = background.shape[:2]
    top_pad_2 = (target_height - current_height) // 2
    bottom_pad_2 = target_height - current_height - top_pad_2
    left_pad_2 = (target_width - current_width) // 2
    right_pad_2 = target_width - current_width - left_pad_2
    background_padding = cv2.copyMakeBorder(background, top_pad_2, bottom_pad_2, left_pad_2, right_pad_2, cv2.BORDER_CONSTANT,
                                            value=(0, 0, 0, 0))
    return foreground_padding, background_padding, (top_pad_1, bottom_pad_1, left_pad_1, right_pad_1), (top_pad_2, bottom_pad_2, left_pad_2, right_pad_2)


def padding_image_1(foreground, background):
    """
    将foreground变成background尺寸一致
    :param foreground:
    :param background:
    :return:
    """
    target_height, target_width = background.shape[:2]
    current_height, current_width = foreground.shape[:2]
    top_pad = (target_height - current_height) // 2
    bottom_pad = target_height - current_height - top_pad
    left_pad = (target_width - current_width) // 2
    right_pad = target_width - current_width - left_pad
    foreground_padding = cv2.copyMakeBorder(foreground, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT,
                                            value=(0, 0, 0, 0))
    return foreground_padding


def rotate_image(img, angle, points=None):
    height, width = img.shape[:2]
    new_size = height + width
    img_empty = np.zeros((new_size, new_size, 4), dtype='uint8')
    img_plane, img_empty, padding_value, _ = padding_image(img, img_empty)
    rotation_matrix = cv2.getRotationMatrix2D((new_size / 2, new_size / 2), angle, 1.0)
    rotated_image = cv2.warpAffine(img_plane, rotation_matrix, (new_size, new_size))
    if points == None:
        return rotated_image
    else:
        points = np.array(points) # 大小n*2
        ones = np.ones((points.shape[0], 1))
        points_with_ones = np.hstack([points + np.array([[padding_value[2], padding_value[0]]]), ones])
        rotated_points = np.dot(rotation_matrix, points_with_ones.T).T
        return rotated_image, rotated_points


def shift_image(img, dx, dy):
    height, width = img.shape[:2]
    new_height, new_width = height + 2 * abs(dy), width + 2 * abs(dx)
    img_new = np.zeros((new_height, new_width, 4), dtype='uint8')
    img_new[(abs(dy) + dy): (abs(dy) + dy + height), (abs(dx) + dx): (abs(dx) + dx + width)] = img
    return img_new


def crop_image(image):
    alpha_channel = image[:, :, 3]
    y_nonzero, x_nonzero = np.nonzero(alpha_channel)
    top, bottom = np.min(y_nonzero), np.max(y_nonzero)
    left, right = np.min(x_nonzero), np.max(x_nonzero)
    cropped_image = image[top:bottom+1, left:right+1]
    return cropped_image, (top, bottom, left, right), (bottom - top, right - left)


def get_mask(img, threshold):
    alpha_channel = img[:, :, 3]
    mask = np.zeros_like(alpha_channel)
    mask[alpha_channel > threshold] = 1
    return mask


def create_mask_with_line(image_shape, center, orient):
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    if orient < 0:
        orient += 360
    if orient == 90 or orient == -90:
        for y in range(image_shape[0]):
            for x in range(image_shape[1]):
                if orient == 90 and x < center[0]:
                    mask[y, x] = 1
        return mask
    rad = np.deg2rad(180-orient) # 相对于x轴和构建的直线斜率而言
    slope = math.tan(rad)
    intercept = center[1] - slope * center[0]
    for y in range(image_shape[0]):
        for x in range(image_shape[1]):
            if ((orient >= 0 and orient < 90) or (orient > 270 and orient <= 360)) and y < slope * x + intercept + 5:
                mask[y, x] = 1
            elif (orient > 90 and orient < 270) and y > slope * x + intercept - 5:
                mask[y, x] = 1
    return mask

import math
if __name__ == "__main__":
    dir = 'processed_data/background_1'
    garage_info_list = load_garage_label(dir, "train")
    plane_info_list = load_plane_label(dir, "train")

    result, location_garage = combine_plane_garage('processed_data/background_1', plane_info_list[0], garage_info_list[0])

    cv2.imwrite('result.png', result)