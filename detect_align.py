from copy import deepcopy
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from jsonargparse import CLI
from tqdm import tqdm

from src.mtcnn_caffe import mtcnn_caffe_detect
from src.mtcnn_onnx import mtcnn_onnx_detect
from src.utils import draw_bboxes_and_keypoints
from src.yolo import yolo_detect

func_dict = {
    "mtcnn_caffe": mtcnn_caffe_detect,
    "mtcnn_onnx": mtcnn_onnx_detect,
    "yolo": yolo_detect,
}

DEBUG = True


def align_face(
    img: np.ndarray,
    src_landmarks: np.ndarray,
    output_size: tuple = (112, 112),
) -> np.ndarray:

    # Check src_landmarks
    assert src_landmarks.shape == (5, 2)
    if src_landmarks.dtype != np.float32:
        src_landmarks.astype(np.float32)

    # Make copy of input image
    src_img = deepcopy(img)

    # Define the standard face template with five landmarks
    # Use template from https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py
    standard_landmarks = np.array(
        [
            [38.2946, 51.6963],  # Left eye
            [73.5318, 51.5014],  # Right eye
            [56.0252, 71.7366],  # Nose tip
            [41.5493, 92.3655],  # Left mouth corner
            [70.7299, 92.2041],  # Right mouth corner
        ],
        dtype=np.float32,
    )

    # Adjust the template for the output size
    if output_size != (112, 112):
        scale_x = output_size[0] / 112.0
        scale_y = output_size[1] / 112.0
        standard_landmarks[:, 0] *= scale_x
        standard_landmarks[:, 1] *= scale_y

    # Estimate the transformation matrix
    transform_matrix = cv2.estimateAffinePartial2D(src_landmarks, standard_landmarks)[0]

    # Apply the transformation to the image
    aligned_img = cv2.warpAffine(
        img,
        transform_matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0),
    )

    if DEBUG:
        # Draw src_landmarks on src_img
        for i in range(len(src_landmarks)):
            cv2.circle(src_img, tuple(src_landmarks[i].astype(int)), 2, (0, 155, 255), 2)

        # Draw standard_landmarks on template_img
        template_img = np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8)
        for i in range(len(standard_landmarks)):
            cv2.circle(template_img, tuple(standard_landmarks[i].astype(int)), 2, (0, 155, 255), 2)

        # Plot all 3 images into one matplotlib plot
        fig, axs = plt.subplots(1, 3, figsize=(10, 4))
        axs: List[plt.Axes]  # Tell IDE the type of axs variable
        axs[0].imshow(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Source Image")
        axs[1].imshow(cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Template")
        axs[2].imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
        axs[2].set_title("Aligned Image")
        fig.suptitle("Face Alignment")
        plt.tight_layout()
        plt.show()

    return aligned_img


def main(
    img_p: str = "data/c-07-twofaces.png",
    out_p: str = "data/c-07-twofaces_aligned.png",
    output_size: tuple = (256, 256),
    method_name: str = "yolo",
    write_images: bool = True,
):
    # assert out_ext in [".png", ".jpg"]
    assert method_name in func_dict.keys()

    # img_paths = sorted(list(data_p.glob("*/*/RGB/*.bmp")))
    error_list = []
    error_count = 0
    # for i in tqdm(range(len(img_paths))):
    # img_p = img_paths[i]
    img_in = cv2.imread(img_p)

    bboxes, keypoints_all = func_dict[method_name](img_in)
    n_detections = len(bboxes)
    img_out = draw_bboxes_and_keypoints(img_in, bboxes, keypoints_all)
    # cv2.imshow("Face Detection", img_out)
    # cv2.waitKey(0)

    if n_detections >= 1:
        # Use biggest bbox
        max_area = 0
        max_idx = 0
        for idx in range(len(bboxes)):
            bbox = bboxes[idx]
            width = bbox[2]
            height = bbox[3]
            area = width * height
            if area > max_area:
                max_area = area
                max_idx = idx

        # Perform face alignment to 112x112 pixel and store aligned images
        keypoints = keypoints_all[max_idx]
        landmarks = np.array(list(keypoints.values()), dtype=np.float32)
        aligned_img = align_face(img_in, landmarks, output_size=output_size)
        if write_images:
            if not cv2.imwrite(out_p, aligned_img):
                raise RuntimeError("Could not write image")
        # cv2.imshow("Aligned Image", aligned_img)
        # cv2.waitKey(0)
    else:
        error_list.append(out_p)
        error_count += 1

    print(f"Detection failed for {error_count} images.")


if __name__ == "__main__":
    CLI(main, as_positional=False)
