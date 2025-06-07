from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import pickle
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
from tools_mine import *
import os
print(os.getcwd())

def fuse_anns(image_background, anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    for ann in sorted_anns:
        m = ann['segmentation']
        if ann['area'] < 5000:
            img[m] = np.concatenate([[255], [255], [255]])
    zoom_factor = (image_background.shape[1] / img.shape[1], image_background.shape[1] / img.shape[1])
    channels = [img[:, :, i] for i in range(img.shape[2])]
    expanded_channels = []
    for channel in channels:
        expanded_channel = zoom(channel, zoom_factor, mode='nearest')
        expanded_channels.append(expanded_channel)
    resized_img = np.stack(expanded_channels, axis=2).astype(np.uint8)
    resized_img_single_channel = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(resized_img_single_channel, 127, 255, cv2.THRESH_BINARY)
    return binary_image

def sign_image(image, mask_image, max_index_w, max_index_h, w_curve, h_curve):
    # 当不传入原始图象时
    image = cv2.cvtColor(mask_image*255, cv2.COLOR_GRAY2BGR)
    range = int(min(max_index_w, w_curve.shape[0] - max_index_w) // 2)
    # 上半截来计算
    _, _, fold_sum2_high = fold_and_sum_with_limit(mask_image, max_index_w, [0, max_index_h], direction='vertical')
    len_fold_length_high = len(fold_sum2_high)
    w_curve_limit_high = np.zeros_like(w_curve)
    w_curve_limit_high[max_index_w - len_fold_length_high: max_index_w] = fold_sum2_high
    w_curve_limit_high[max_index_w: max_index_w + len_fold_length_high] = fold_sum2_high[::-1]
    # 下半截来计算
    _, _, fold_sum2_low = fold_and_sum_with_limit(mask_image, max_index_w, [max_index_h, mask_image.shape[0]],
                                                  direction='vertical')
    len_fold_length_low = len(fold_sum2_low)
    w_curve_limit_low = np.zeros_like(w_curve)
    w_curve_limit_low[max_index_w - len_fold_length_low: max_index_w] = fold_sum2_low
    w_curve_limit_low[max_index_w: max_index_w + len_fold_length_low] = fold_sum2_low[::-1]

    # 机头
    w_curve_diff_high = smooth_and_differentiate(w_curve_limit_high, 11, 5)
    w_curve_diff1_high = smooth_and_differentiate(w_curve_diff_high, 11, 5)
    index_w_1_high = np.argmax(w_curve_diff1_high[max_index_w - range: max_index_w]) + (max_index_w - range)
    index_w_2_high = 2 * max_index_w - index_w_1_high
    cv2.line(image, [index_w_1_high, 0], [index_w_1_high, max_index_h], (0, 0, 255), 1)
    cv2.line(image, [index_w_2_high, 0], [index_w_2_high, max_index_h], (0, 0, 255), 1)

    # 机翼
    # h_curve_diff = smooth_and_differentiate(h_curve, 3, 1)
    # index_h_1 = np.argmax(h_curve_diff[0: max_index_h])
    # index_h_2 = np.argmin(h_curve_diff[max_index_h:]) + max_index_h
    # cv2.line(image, [0, index_h_1], [image.shape[1], index_h_1], (0, 255, 0), 1)
    # cv2.line(image, [0, index_h_2], [image.shape[1], index_h_2], (0, 255, 0), 1)

    h_curve_diff = smooth_and_differentiate(h_curve, 3, 1)
    h_curve_diff1 = smooth_and_differentiate(h_curve_diff, 3, 1)
    index_h_once_1 = np.argmax(h_curve_diff[0: max_index_h])
    index_h_once_2 = np.argmin(h_curve_diff[max_index_h:]) + max_index_h
    slice_win = 3 if (h_curve.shape[0] / 20) < 2 else 2
    index_h_1 = np.argmax(h_curve_diff1[0: max_index_h]) if index_h_once_1 > slice_win else index_h_once_1
    index_h_2 = np.argmax(h_curve_diff1[max_index_h:]) + max_index_h if index_h_once_2 < (h_curve.shape[0] - slice_win) else index_h_once_2
    cv2.line(image, [0, index_h_1], [image.shape[1], index_h_1], (0, 255, 0), 1)
    cv2.line(image, [0, index_h_2], [image.shape[1], index_h_2], (0, 255, 0), 1)

    # h_up_min_index = np.argmin(h_curve[0: max_index_h])
    # h_down_min_index = np.argmin(h_curve[max_index_h:]) + max_index_h
    # h_curve_diff = smooth_and_differentiate(h_curve, 3, 1)
    # h_curve_diff1 = smooth_and_differentiate(h_curve_diff, 3, 1)
    # print(h_up_min_index, max_index_h, h_down_min_index)
    # index_h_1 = np.argmax(h_curve_diff1[h_up_min_index: max_index_h]) if h_up_min_index != max_index_h else max_index_h
    # index_h_2 = np.argmax(h_curve_diff1[max_index_h:h_down_min_index]) + max_index_h if max_index_h != h_down_min_index else max_index_h
    # cv2.line(image, [0, index_h_1], [image.shape[1], index_h_1], (0, 255, 0), 2)
    # cv2.line(image, [0, index_h_2], [image.shape[1], index_h_2], (0, 255, 0), 2)

    # 机尾
    w_curve_diff_low = smooth_and_differentiate(w_curve_limit_low, 11, 5)
    w_curve_diff_low1 = smooth_and_differentiate(w_curve_diff_low, 11, 5)
    index_w_1_low = np.argmax(w_curve_diff_low1[max_index_w - range: max_index_w]) + (max_index_w - range)
    index_w_2_low = 2 * max_index_w - index_w_1_low
    cv2.line(image, [index_w_1_low, max_index_h], [index_w_1_low, image.shape[0]], (0, 0, 255), 1)
    cv2.line(image, [index_w_2_low, max_index_h], [index_w_2_low, image.shape[0]], (0, 0, 255), 1)

    box_part = dict()
    box_part['head'] = np.array([[index_w_1_high, 0], [index_w_2_high, 0], [index_w_2_high, max_index_h], [index_w_1_high, max_index_h]])
    box_part['wing'] = np.array([[0, index_h_1], [image.shape[1], index_h_1], [image.shape[1], index_h_2], [0, index_h_2]])
    box_part['tail'] = np.array([[index_w_1_low, max_index_h], [index_w_2_low, max_index_h], [index_w_2_low, image.shape[0]], [index_w_1_low, image.shape[0]]])

    return image, mask_image, w_curve_diff_low1, h_curve_diff, box_part

def sign_image2(image, mask_image, max_index_w, max_index_h, w_curve, h_curve):
    # 当不传入原始图象时
    image = cv2.cvtColor(mask_image*255, cv2.COLOR_GRAY2BGR)
    # 下半截来计算
    _, _, fold_sum2_low = fold_and_sum_with_limit(mask_image, max_index_w, [max_index_h, mask_image.shape[0]],
                                                  direction='vertical')
    len_fold_length = len(fold_sum2_low)
    w_curve_limit = np.zeros_like(w_curve)
    w_curve_limit[max_index_w - len_fold_length: max_index_w] = fold_sum2_low
    w_curve_limit[max_index_w: max_index_w + len_fold_length] = fold_sum2_low[::-1]


    # 机头
    w_curve_diff = smooth_and_differentiate(w_curve, 11, 5)
    w_curve_diff1 = smooth_and_differentiate(w_curve_diff, 11, 5)
    index_w_1 = np.argmax(w_curve_diff1)
    index_w_2 = 2 * max_index_w - index_w_1
    cv2.line(image, [index_w_1, 0], [index_w_1, max_index_h], (0, 0, 255), 1)
    cv2.line(image, [index_w_2, 0], [index_w_2, max_index_h], (0, 0, 255), 1)

    # 机翼
    h_curve_diff = smooth_and_differentiate(h_curve, 3, 1)
    h_curve_diff1 = smooth_and_differentiate(h_curve_diff, 3, 1)
    index_h_1 = np.argmax(h_curve_diff1[0: max_index_h])
    index_h_2 = np.argmax(h_curve_diff1[max_index_h:]) + max_index_h
    cv2.line(image, [0, index_h_1], [image.shape[1], index_h_1], (0, 255, 0), 1)
    cv2.line(image, [0, index_h_2], [image.shape[1], index_h_2], (0, 255, 0), 1)

    # 机尾
    w_curve_diff_low = smooth_and_differentiate(w_curve_limit, 11, 5)
    w_curve_diff_low1 = smooth_and_differentiate(w_curve_diff_low, 11, 5)
    index_w_1_low = np.argmax(w_curve_diff_low1)
    index_w_2_low = 2 * max_index_w - index_w_1_low
    cv2.line(image, [index_w_1_low, max_index_h], [index_w_1_low, image.shape[0]], (0, 0, 255), 1)
    cv2.line(image, [index_w_2_low, max_index_h], [index_w_2_low, image.shape[0]], (0, 0, 255), 1)

    return image, mask_image, w_curve_diff_low1, h_curve_diff1

if __name__ == "__main__":
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda:0"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)
    image = cv2.imread('test/4.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks1 = mask_generator.generate(image)
    predictor.set_image(image)
    predictor.reset_image()
    with open('inter_data/mask_test.pkl', 'wb') as f:
        pickle.dump(masks1, f)
    #
    # [390, 1125]
    # 40度
    # h:210 w:200


    # image = cv2.imread('test/plane.jpg')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # with open('inter_data/mask_test.pkl', 'rb') as f:
    #     masks1 = pickle.load(f)

    mask_image = fuse_anns(image, masks1)

    scores_h, scores_w, fold_sum_h, fold_sum_w, fold_sum2_h, fold_sum2_w = symmetry_caculate(mask_image)
    max_index_w, max_index_h, w_curve, h_curve = get_curve_plus(scores_h, scores_w, fold_sum_h, fold_sum_w, fold_sum2_h, fold_sum2_w)

    image, mask_image, w_curve_diff_low1, h_curve_diff1, _ = sign_image(image, mask_image, max_index_w, max_index_h, w_curve, h_curve)
    plot_image(mask_image, w_curve_diff_low1, h_curve_diff1)