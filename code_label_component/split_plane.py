import cv2
import numpy as np
import matplotlib.pyplot as plt

def crop_rotated_rectangle(image, x, y, w, h, rad):
    # Convert angle to radians
    angle = np.rad2deg(rad)

    # Get the coordinates of the rotated rectangle's four vertices
    rect = ((x, y), (w, h), angle)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Find the bounding box of the rotated rectangle
    x_min = min(box[:, 0])
    x_max = max(box[:, 0])
    y_min = min(box[:, 1])
    y_max = max(box[:, 1])
    x_min_ = max(0, x_min)
    x_max_ = min(image.shape[1], x_max)
    y_min_ = max(0, y_min)
    y_max_ = min(image.shape[0], y_max)

    # Calculate the width and height of the bounding box
    width = x_max_ - x_min_
    height = y_max_ - y_min_

    # # Translate the rotation matrix to the top-left corner of the bounding box
    # rotation_matrix[0, 2] += width / 2 - x
    # rotation_matrix[1, 2] += height / 2 - y

    # Perform the affine transformation to get the rotated rectangle
    rotation_matrix = cv2.getRotationMatrix2D((x-x_min_, y-y_min_), angle, 1.0)
    rotated_image = cv2.warpAffine(image[y_min_: y_max_, x_min_: x_max_], rotation_matrix, (2*int(w), 2*int(h)))
    # rotated_image = cv2.warpAffine(image[y_min_: y_max_, x_min_: x_max_], rotation_matrix, (width, height))

    # ones = np.ones((4, 1))
    # box_with_ones = np.hstack([box - np.array([x_min, y_min]), ones])
    # # Apply the rotation matrix to each point
    # rotated_box = np.dot(rotation_matrix, box_with_ones.T).T
    # x_min_r = int(max(min(rotated_box[:, 0]), 0))
    # x_max_r = int(min(max(rotated_box[:, 0]), width))
    # y_min_r = int(max(min(rotated_box[:, 1]), 0))
    # y_max_r = int(min(max(rotated_box[:, 1]), height))
    #
    # # # Crop the bounding box
    # cropped_image = rotated_image[y_min_r:y_max_r, x_min_r:x_max_r]

    return image[y_min: y_max, x_min: x_max], rotated_image

def crop_rotated_rectangle2(image, x, y, w, h, rad, out_color_image = False):
    # Convert angle to radians
    angle = np.rad2deg(rad)

    # Get the coordinates of the rotated rectangle's four vertices
    rect = ((x, y), (w, h), angle)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Find the bounding box of the rotated rectangle
    x_min = min(box[:, 0])
    x_max = max(box[:, 0])
    y_min = min(box[:, 1])
    y_max = max(box[:, 1])
    width = x_max - x_min
    height = y_max - y_min
    # 既要保证开始时能取到所有飞机，还要保证旋转后能包含所有飞机
    ori_size = max(width, height, h, w)
    x, y = int(x), int(y)
    if ori_size % 2 == 1:
        ori_size += 1
    # int(ori_size/2)不会影像对应关系，因为基准是中心点
    half_ori_size = int(ori_size/2)
    if len(image.shape) == 2:
        ori_image = np.zeros((ori_size + 1, ori_size + 1), dtype='uint8')
    else:
        ori_image = np.zeros((ori_size + 1, ori_size + 1, 3), dtype='uint8')
    ori_x1, ori_x2 = x - half_ori_size, x + half_ori_size
    ori_y1, ori_y2 = y - half_ori_size, y + half_ori_size
    ori_x1 = max(0, ori_x1)
    ori_x2 = min(image.shape[1], ori_x2)
    ori_y1 = max(0, ori_y1)
    ori_y2 = min(image.shape[0], ori_y2)
    ori_image[half_ori_size-(y-ori_y1): half_ori_size+(ori_y2-y), half_ori_size-(x-ori_x1): half_ori_size+(ori_x2-x)] = \
        image[ori_y1: ori_y2, ori_x1: ori_x2]

    # Perform the affine transformation to get the rotated rectangle
    rotation_matrix = cv2.getRotationMatrix2D((ori_size / 2, ori_size / 2), angle, 1.0)
    rotated_image = cv2.warpAffine(ori_image, rotation_matrix, (ori_size, ori_size))
    # int(h / 2)不会影像对应关系，因为基准是中心点
    cropped_image = rotated_image[half_ori_size - int(h / 2): half_ori_size + int(h / 2), half_ori_size - int(w / 2): half_ori_size + int(w / 2)]
    if out_color_image == False:
        _, cropped_image = cv2.threshold(cropped_image, 127, 255, cv2.THRESH_BINARY)

    # ones = np.ones((4, 1))
    # box_with_ones = np.hstack([box - np.array([x_min, y_min]), ones])
    # # Apply the rotation matrix to each point
    # rotated_box = np.dot(rotation_matrix, box_with_ones.T).T

    off_scale_image2crop = np.array([x - half_ori_size, y - half_ori_size]).reshape(1, 2)
    off_scale_rotate2crop = np.array([half_ori_size - int(w / 2), half_ori_size - int(h / 2)]).reshape(1, 2)
    return ori_image, cropped_image, [off_scale_image2crop, off_scale_rotate2crop]

def rotated_boxes_parts(box_part, off_scale, origin_size, rad):

    angle = np.rad2deg(rad)
    rotation_matrix = cv2.getRotationMatrix2D((origin_size / 2, origin_size / 2), -angle, 1.0)
    ones = np.ones((4, 1))
    boxes_on_image = {}
    for (part, box_) in box_part.items():
        box = box_ + off_scale[1]
        box_with_ones = np.hstack([box, ones])
        rotated_box = np.dot(rotation_matrix, box_with_ones.T).T
        rotated_box_on_image = rotated_box + off_scale[0]
        boxes_on_image[part] = rotated_box_on_image
    return boxes_on_image


if __name__=="__main__":
    # Example usage
    image = cv2.imread('/root/autodl-tmp/mmrotate/data/DOTA/train/images/P0036.png')
    x, y, w, h, angle = 2322.969482421875, 126.20022583007812, 177.0113067626953, 123.94690704345703, -1.559497373809612
    # x, y, w, h, angle = 1530.9718017578125, 3922.66357421875, 167.56283569335938, 134.33164978027344, 0.5136060958456171
    ori_image, cropped_image, off_scale = crop_rotated_rectangle2(image, x, y, w, h, angle, out_color_image = True)
    box_points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)
    box_part = {}
    box_part['head'] = np.array([[0, 0], [100.5, 0], [100.5, 100.5], [0, 100.5]])
    cv2.drawContours(cropped_image, [box_points], 0, (0, 255, 0), 2)
    boxes = rotated_boxes_parts(box_part, off_scale, ori_image.shape[0]-1, angle)
    boxes_draw = boxes['head'].astype(int)
    cv2.drawContours(image, [boxes_draw], 0, (0, 255, 0), 2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 15))
    ax1.imshow(ori_image)
    ax1.axis('off')
    ax2.imshow(cropped_image)
    ax2.axis('off')
    plt.show()

    plt.imshow(image[0:500, 2000: 2500])
    plt.show()
