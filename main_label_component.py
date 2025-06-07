import os
import os.path as osp
import matplotlib.pyplot as plt
import mmcv
import cv2
import numpy as np
from PIL import Image
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
import random
import math
import numpy as np
from .code_label_component.split_plane import *
from .code_label_component.image_sign import *

dota_img_path = '/data/liuyi/pyproject/mmrotate/data/split_ss_dota/train/images/'
dota_anns_file = "/data/liuyi/pyproject/mmrotate/data/split_ss_dota/train/annfiles/"

def crop_boxes(boxes, ks):
    n = boxes.shape[0]
    v = len(ks)
    cropped_boxes = np.zeros((n, v, 5))

    for i in range(n):
        x, y, w, h, theta = boxes[i]
        for j, k in enumerate(ks):
            new_w = w / k
            new_h = h / k

            if random.random() < 0.5:
                # 变动x
                dx = (new_w - w) / 2 * np.cos(theta)
                dy = (new_w - w) / 2 * np.sin(theta)
            else:
                # 变动y
                dx = -(new_h - h) / 2 * np.sin(theta)
                dy = (new_h - h) / 2 * np.cos(theta)

            if random.random() < 0.25:
                # 变动x
                new_x = x + dx
                new_y = y + dy
            elif random.random() < 0.5:
                new_x = x - dx
                new_y = y + dy
            elif random.random() < 0.75:
                new_x = x + dx
                new_y = y - dy
            else:
                new_x = x - dx
                new_y = y - dy

            cropped_boxes[i, j] = [new_x, new_y, new_w, new_h, theta]

    return cropped_boxes

def plot_image_boxes(img_name, boxes):
    img = mmcv.imread(dota_img_path + img_name)
    # 绘制旋转边界框
    for i, box in enumerate(boxes):
        cx, cy, w, h, theta = box

        # 计算旋转矩形的角点
        rect = ((cx, cy), (w, h), np.degrees(theta))  # OpenCV 需要角度是度数
        box_points = cv2.boxPoints(rect)  # 获取旋转矩形的四个顶点
        box_points = np.int0(box_points)  # 转换为整数

        # 在图像上绘制旋转边界框
        cv2.drawContours(img, [box_points], 0, (0, 255, 0), 2)  # 绿色边框

    # 显示图像
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 转换颜色为 RGB 显示
    plt.axis('off')  # 隐藏坐标轴
    plt.show()
    plt.savefig('trash/split_result.png', bbox_inches='tight')

if __name__ == "__main__":
    cls_map = {c: i for i, c in enumerate(['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter'])}
    anns_file = dota_anns_file

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda:0"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)

    i = 0
    for ann_file in os.listdir(anns_file):
        data_info = {}
        img_id = osp.split(ann_file)[1][:-4]
        img_name = img_id + '.png'
        img = mmcv.imread(dota_img_path + img_name)
        data_info['filename'] = img_name
        data_info['ann'] = {}
        gt_bboxes = []
        gt_labels = []
        gt_polygons = []
        gt_bboxes_ignore = []
        gt_labels_ignore = []
        gt_polygons_ignore = []

        with open(anns_file + ann_file) as file1:
            file1.seek(0)
            if(len(file1.readlines())==2):
                continue
            file1.seek(0)
            lines_ = file1.readlines()
            sign = 0
            num_plane = 0
            for line in lines_[2:]:
                values = line.split()
                if values[-2] == "plane":
                    sign = 1
                    num_plane += 1
            if sign == 0:
                continue
            print(num_plane)

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masks1 = mask_generator.generate(image)
        predictor.set_image(image)
        predictor.reset_image()
        with open('test/mask_test.pkl', 'wb') as f:
            pickle.dump(masks1, f)
        mask_image = fuse_anns(image, masks1)
        print("got one mask")

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 15))
        # ax1.imshow(image)
        # ax1.axis('off')
        # ax2.imshow(mask_image)
        # ax2.axis('off')
        # # plt.savefig('test/out4.jpg')
        # plt.show()

        with open(anns_file + ann_file) as f:
            f.seek(0)
            s = f.readlines()[2:]
            for si in s:
                bbox_info = si.split()
                if bbox_info[-2] == "plane":
                    poly = np.array(bbox_info[:8], dtype=np.float32)
                    try:
                        x, y, w, h, a = poly2obb_np(poly, 'le90')
                    except:  # noqa: E722
                        continue
                    cls_name = bbox_info[8]
                    difficulty = int(bbox_info[9])
                    label = cls_map[cls_name]
                    if difficulty > 100:
                        pass
                    else:
                        gt_bboxes.append([x, y, w, h, a])
                        gt_labels.append(label)
                        gt_polygons.append(poly)

                    i = i+1
                    ori_image, cropped_image, off_scale = crop_rotated_rectangle2(mask_image, x, y, w, h, a)
                    scores_h, scores_w, fold_sum_h, fold_sum_w, fold_sum2_h, fold_sum2_w = symmetry_caculate(cropped_image)
                    max_value_h, max_value_w = max(scores_h), max(scores_w)
                    if max_value_h > max_value_w:
                        print('rotate')
                        plt.imshow(cropped_image)
                        # plt.imshow(ori_image)
                        plt.show()
                        a = a + math.pi
                        ori_image, cropped_image, off_scale = crop_rotated_rectangle2(mask_image, x, y, w, h, a)
                        scores_h, scores_w, fold_sum_h, fold_sum_w, fold_sum2_h, fold_sum2_w = symmetry_caculate(cropped_image)

                    max_index_w, max_index_h, w_curve, h_curve = get_curve_plus(scores_h, scores_w, fold_sum_h,
                                                                                fold_sum_w, fold_sum2_h, fold_sum2_w)

                    if max_index_h > 1 and max_index_w > 1 and max_index_h < len(h_curve)-1 and max_index_w < len(w_curve)-1:
                        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 15))
                        # ax1.imshow(ori_image)
                        # ax1.axis('off')
                        # ax2.imshow(cropped_image)
                        # ax2.axis('off')
                        # # plt.savefig('test/result_bin.jpg')
                        # plt.show()

                        cropped_image2 = cropped_image
                        image_single, mask_image_single, w_curve_diff_low1, h_curve_diff1, boxes_part = sign_image(cropped_image, cropped_image, max_index_w, max_index_h, w_curve, h_curve)
                        boxes_part_rotate = rotated_boxes_parts(boxes_part, off_scale, ori_image.shape[0] - 1, a)
                        # plot_image(image_single, w_curve, h_curve, w_curve, h_curve_diff1)

                        # image2, mask_image2, w_curve_diff_low12, h_curve_diff12 = sign_image2(cropped_image2,
                        #                                                                       cropped_image2,
                        #                                                                       max_index_w, max_index_h,
                        #                                                                       w_curve, h_curve)
                        # plot_image(image2, w_curve, h_curve, w_curve, h_curve_diff12)
                        boxes_draw = boxes_part_rotate['head'].astype(int)
                        cv2.drawContours(image, [boxes_draw], 0, (0, 255, 0), 2)
                        boxes_draw = boxes_part_rotate['wing'].astype(int)
                        cv2.drawContours(image, [boxes_draw], 0, (255, 0, 0), 2)
                        boxes_draw = boxes_part_rotate['tail'].astype(int)
                        cv2.drawContours(image, [boxes_draw], 0, (0, 0, 255), 2)
            fig = plt.figure(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
            plt.imshow(image)
            plt.axis('off')
            plt.savefig('test/part_crop/' + img_id + '_part.png', dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            # ks = [3, 3, 4, 3, 2]
            # bboxes_ann = np.array(gt_bboxes, dtype=np.float32)
            # # # 之后要将其中的部件也crop，作为数据增强的训练数据
            # cropped_boxes = crop_boxes(bboxes_ann, ks).reshape(-1, 5)
            # plot_image_boxes(img_name, cropped_boxes)


