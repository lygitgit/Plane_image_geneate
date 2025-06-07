import cv2
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def get_anns(image_background, anns, ax, indices=None):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))
    for ann in sorted_anns:
        m = ann['segmentation']
        img[m] = 1

    zoom_factor = (image_background.shape[1] / img.shape[1], image_background.shape[1] / img.shape[1])
    channels = [img[:, :, i] for i in range(img.shape[2])]
    expanded_channels = []
    for channel in channels:
        expanded_channel = zoom(channel, zoom_factor, mode='nearest')
        expanded_channels.append(expanded_channel)
    resized_img = np.stack(expanded_channels, axis=2)
    return resized_img

def fold_and_sum(image, axis_pos, direction='horizontal'):
    h, w = image.shape
    image[image==255] = 1
    fold_sum2 = []
    score = 0
    if direction == 'horizontal':
        fold_sum = np.zeros((w))
        # 上半部分折叠到下半部分
        for i in range(h):
            if i < axis_pos:
                if (2 * axis_pos - i) < h:
                    score += np.sum(image[2 * axis_pos - i, :] * image[i, :])
                    fold_sum += image[2 * axis_pos - i, :] * image[i, :]
                    fold_sum2.append(np.sum(image[2 * axis_pos - i, :] * image[i, :]))
    else:  # vertical
        # 左半部分折叠到右半部分
        fold_sum = np.zeros((h))
        for j in range(w):
            if j < axis_pos:
                if (2*axis_pos - j) < w:
                    score += np.sum(image[:, 2*axis_pos - j] * image[:, j])
                    fold_sum += image[:, 2*axis_pos - j] * image[:, j]
                    fold_sum2.append(np.sum(image[:, 2*axis_pos - j] * image[:, j]))
    return score, fold_sum, np.array(fold_sum2)

def fold_and_sum_with_limit(image, axis_pos, limit, direction='horizontal'):
    h, w = image.shape
    image[image==255] = 1
    score = 0
    fold_sum2 = []
    if direction == 'horizontal':
        fold_sum = np.zeros((w))
        # 上半部分折叠到下半部分
        for i in range(h):
            if i < axis_pos:
                if (2 * axis_pos - i) < h:
                    score += np.sum(image[2 * axis_pos - i, :] * image[i, :])
                    fold_sum[limit[0]:limit[1]] += image[2 * axis_pos - i, limit[0]:limit[1]] * image[i, limit[0]:limit[1]]
                    fold_sum2.append(np.sum(image[2 * axis_pos - i, limit[0]:limit[1]] * image[i, limit[0]:limit[1]]))
    else:  # vertical
        # 左半部分折叠到右半部分
        fold_sum = np.zeros((h))
        for j in range(w):
            if j < axis_pos:
                if (2*axis_pos - j) < w:
                    score += np.sum(image[:, 2*axis_pos - j] * image[:, j])
                    fold_sum[limit[0]:limit[1]] += image[limit[0]:limit[1], 2*axis_pos - j] * image[limit[0]:limit[1], j]
                    fold_sum2.append(np.sum(image[limit[0]:limit[1], 2 * axis_pos - j] * image[limit[0]:limit[1], j]))
    return score, fold_sum, np.array(fold_sum2)

def symmetry_caculate(binary_image):
    # 设置取样间隔
    interval = 1
    scores_h = []
    scores_w = []
    fold_sum_h = []
    fold_sum_w = []
    fold_sum2_h = []
    fold_sum2_w = []
    # 遍历横向轴
    for pos in range(0, binary_image.shape[0], interval):
        score, fold_sum, fold_sum2 = fold_and_sum(binary_image, pos, direction='horizontal')
        # print(f"Horizontal axis at position {pos}: Sum: {score}")
        scores_h.append(score)
        fold_sum_h.append(fold_sum)
        fold_sum2_h.append(fold_sum2)

    # 遍历纵向轴
    for pos in range(0, binary_image.shape[1], interval):
        score, fold_sum, fold_sum2 = fold_and_sum(binary_image, pos, direction='vertical')
        # print(f"Vertical axis at position {pos}: Sum: {score}")
        scores_w.append(score)
        fold_sum_w.append(fold_sum)
        fold_sum2_w.append(fold_sum2)

    return scores_h, scores_w, fold_sum_h, fold_sum_w, fold_sum2_h, fold_sum2_w

from scipy.signal import savgol_filter
def smooth_and_differentiate(arr, window_length, polyorder, derivative=1):
    smoothed_arr = savgol_filter(arr, window_length, polyorder)
    differentiated_arr = np.gradient(smoothed_arr)
    slide_win = int(arr.shape[0] / 10) if int(arr.shape[0] / 10) % 2 != 0 else int(arr.shape[0] / 10) + 1
    slide_win = 3 if slide_win < 3 else slide_win
    array = sliding_window_sum(differentiated_arr, slide_win)
    return array

def sliding_window_sum(arr, window_size):
    # Ensure the window size is odd
    if window_size % 2 == 0:
        raise ValueError("Window size should be odd")
    half_window = window_size // 2
    result = np.zeros_like(arr)
    # 填充数组，处理边界问题
    pad_arr = np.pad(arr, (half_window, half_window), mode='edge')
    for i in range(len(arr)):
        start = i
        end = i + window_size
        result[i] = np.sum(pad_arr[start:end])
    return result

def get_curve_plus(scores_h, scores_w, fold_sum_h, fold_sum_w, fold_sum2_h, fold_sum2_w):
    arr_h = np.array(scores_h)
    max_value_h = np.max(arr_h)
    max_index_h = np.argmax(arr_h)
    # cv2.line(image, [0, max_index_h], [image.shape[1], max_index_h], (0, 255, 0), 2)

    arr_w = np.array(scores_w)
    max_value_w = np.max(arr_w)
    max_index_w = np.argmax(arr_w)
    # cv2.line(image, [max_index_w, 0], [max_index_w, image.shape[0]], (0, 0, 255), 2)
    # w_curve无意义，因为飞机前后不对称
    w_curve = fold_sum_h[max_index_h]  # fold_sum_h代表list长度为h，是在h上逐个进行的水平翻折
    h_curve = fold_sum_w[max_index_w]
    w_curve_vertical = np.zeros_like(w_curve)
    len_fold_length = len(fold_sum2_w[max_index_w])
    w_curve_vertical[max_index_w - len_fold_length: max_index_w] = fold_sum2_w[max_index_w]
    w_curve_vertical[max_index_w: max_index_w + len_fold_length] = fold_sum2_w[max_index_w][::-1]

    return max_index_w, max_index_h, w_curve_vertical, h_curve

def plot_image(image, w_curve1, h_curve1, w_curve2=None, h_curve2=None):
    longer_side = np.max(image.shape)
    longer_side_index = np.argmax(image.shape)
    w_norm, h_norm = image.shape[0] / longer_side, image.shape[1] / longer_side
    w_axis = np.linspace(0, image.shape[1] - 1, image.shape[1])
    h_axis = np.linspace(0, image.shape[0] - 1, image.shape[0])
    # 创建一个包含两个子图的网格布局
    fig = plt.figure(figsize=(10 * h_norm, 10 * w_norm))
    if w_curve2 is None and h_curve2 is None:
        gs = GridSpec(2, 2, figure=fig, width_ratios=[0.1, 0.9], height_ratios=[0.1, 0.9])
        main_ax = fig.add_subplot(gs[1, 1])
        top_ax = fig.add_subplot(gs[0, 1], sharex=main_ax)
        left_ax = fig.add_subplot(gs[1, 0], sharey=main_ax)
    else:
        gs = GridSpec(3, 3, figure=fig, width_ratios=[0.1, 0.1, 0.8], height_ratios=[0.1, 0.1, 0.8])
        main_ax = fig.add_subplot(gs[2, 2])
        top_ax = fig.add_subplot(gs[0, 2], sharex=main_ax)
        left_ax = fig.add_subplot(gs[2, 0], sharey=main_ax)

    # 在主图上显示图像
    main_ax.imshow(image)
    main_ax.axis('off')

    # 在图像上方创建顶部曲线

    top_ax.plot(w_axis, w_curve1, color='blue')
    top_ax.xaxis.set_label_position('top')
    top_ax.xaxis.tick_top()
    top_ax.set_yticks([])
    top_ax.set_xticks([])

    # 在图像左侧创建左侧曲线

    left_ax.plot(h_curve1, h_axis, color='red')
    left_ax.yaxis.set_label_position('left')
    left_ax.yaxis.tick_left()
    left_ax.invert_xaxis()
    left_ax.set_xticks([])
    left_ax.set_yticks([])

    if w_curve2 is not None and h_curve2 is not None:
        # 在图像上方创建顶部曲线
        top_ax = fig.add_subplot(gs[1, 2], sharex=main_ax)
        top_ax.plot(w_axis, w_curve2, color='blue')
        top_ax.xaxis.set_label_position('top')
        top_ax.xaxis.tick_top()
        top_ax.set_yticks([])
        top_ax.set_xticks([])

        # 在图像左侧创建左侧曲线
        left_ax = fig.add_subplot(gs[2, 1], sharey=main_ax)
        left_ax.plot(h_curve2, h_axis, color='red')
        left_ax.yaxis.set_label_position('left')
        left_ax.yaxis.tick_left()
        left_ax.invert_xaxis()
        left_ax.set_xticks([])
        left_ax.set_yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = 'trash/mask.jpg'
    image = cv2.imread(image_path)
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
    scores_h, scores_w, fold_sum_h, fold_sum_w, fold_sum2_h, fold_sum2_w = symmetry_caculate(binary_image)
    max_index_w, max_index_h, w_curve, h_curve = get_curve_plus(scores_h, scores_w, fold_sum_h, fold_sum_w, fold_sum2_h, fold_sum2_w)

    cv2.line(image, [0, max_index_h], [image.shape[1], max_index_h], (0, 255, 0), 2)
    cv2.line(image, [max_index_w, 0], [max_index_w, image.shape[0]], (0, 0, 255), 2)

    # 下半截来计算
    _, _, fold_sum2_low = fold_and_sum_with_limit(binary_image, max_index_w, [max_index_h, binary_image.shape[0]], direction='vertical')
    len_fold_length = len(fold_sum2_low)
    w_curve_limit = np.zeros_like(w_curve)
    w_curve_limit[max_index_w - len_fold_length: max_index_w] = fold_sum2_low
    w_curve_limit[max_index_w: max_index_w + len_fold_length] = fold_sum2_low[::-1]

    w_curve_diff = smooth_and_differentiate(w_curve, 10, 5)
    w_curve_diff1 = smooth_and_differentiate(w_curve_diff, 10, 5)
    index_w_1 = np.argmax(w_curve_diff1)
    index_w_2 = 2 * max_index_w - index_w_1
    cv2.line(image, [index_w_1, 0], [index_w_1, max_index_h], (0, 0, 255), 2)
    cv2.line(image, [index_w_2, 0], [index_w_2, max_index_h], (0, 0, 255), 2)

    h_curve_diff = smooth_and_differentiate(h_curve, 10, 5)
    h_curve_diff1 = smooth_and_differentiate(h_curve_diff, 10, 5)
    index_h_1 = np.argmax(h_curve_diff1[0: max_index_h])
    index_h_2 = np.argmax(h_curve_diff1[max_index_h:]) + max_index_h
    cv2.line(image, [0, index_h_1], [image.shape[1], index_h_1], (0, 255, 0), 2)
    cv2.line(image, [0, index_h_2], [image.shape[1], index_h_2], (0, 255, 0), 2)

    w_curve_diff_low = smooth_and_differentiate(w_curve_limit, 10, 5)
    w_curve_diff_low1 = smooth_and_differentiate(w_curve_diff_low, 10, 5)
    index_w_1_low = np.argmax(w_curve_diff_low1)
    index_w_2_low = 2 * max_index_w - index_w_1_low
    cv2.line(image, [index_w_1_low, max_index_h], [index_w_1_low, image.shape[0]], (0, 0, 255), 2)
    cv2.line(image, [index_w_2_low, max_index_h], [index_w_2_low, image.shape[0]], (0, 0, 255), 2)

    plot_image(image, w_curve, h_curve, w_curve_diff_low1, h_curve_diff1)

