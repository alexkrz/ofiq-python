from copy import deepcopy

import cv2
import numpy as np


def draw_bboxes_and_keypoints(
    img: np.ndarray, bboxes: list, keypoints_all: list | None = None
) -> np.ndarray:
    img = deepcopy(img)
    for i in range(len(bboxes)):
        # bbox is a list of four integers (x, y, w, h)
        bbox = bboxes[i]
        x, y, w, h = bbox
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 155, 255),
            2,
        )

        if keypoints_all is not None:
            # keypoints is a dictionary of the five facial landmarks with their corresponding (x, y) coordinates
            keypoints = keypoints_all[i]
            cv2.circle(img, (keypoints["left_eye"]), 2, (0, 155, 255), 2)
            cv2.circle(img, (keypoints["right_eye"]), 2, (0, 155, 255), 2)
            cv2.circle(img, (keypoints["nose"]), 2, (0, 155, 255), 2)
            cv2.circle(img, (keypoints["mouth_left"]), 2, (0, 155, 255), 2)
            cv2.circle(img, (keypoints["mouth_right"]), 2, (0, 155, 255), 2)
    return img


def crop_and_resize(img: np.ndarray, bbox: np.ndarray, output_size: tuple) -> np.ndarray:
    x, y, width, height = bbox.tolist()

    # Extend bounding box to square
    if width < height:
        diff = height - width
        width = height
        x = (int)(x - diff / 2)
    elif height < width:
        diff = width - height
        height = width
        y = (int)(y - diff / 2)

    # bbox = np.array([x, y, width, height]).astype("int")
    # print("bbox_square:", bbox)
    # img_out = draw_bboxes_and_keypoints(img, [bbox], keypoints_all=None)
    # cv2.imshow("Face Detection", img_out)
    # cv2.waitKey(0)

    # Add padding
    # NOTE: We perform padding here different to the ISO standard
    padding = (int)(0.1 * width)
    width = width + padding
    height = height + padding
    x = (int)(x - padding / 2)
    y = (int)(y - padding / 2)

    # Crop image
    img = img[y : y + height, x : x + width]
    # cv2.imshow("Cropped img", img)
    # cv2.waitKey(0)

    # Resize to output_size
    img_out = cv2.resize(img, dsize=output_size)
    return img_out
