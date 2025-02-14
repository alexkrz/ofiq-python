import cv2
import matplotlib.pyplot as plt
import numpy as np
from jsonargparse import CLI
from tqdm import tqdm

from src.ssd import ssd_detect
from src.utils import crop_and_resize, draw_bboxes_and_keypoints

func_dict = {
    "ssd": ssd_detect,
}


def main(
    img_p: str = "data/009_03.jpg",
    out_p: str = "data/009_03_cropped.png",
    output_size: tuple = (616, 616),
    method_name: str = "ssd",
    write_images: bool = True,
):
    assert method_name in func_dict.keys()

    img_in = cv2.imread(img_p)
    bboxes = ssd_detect(img_in)
    n_detections = len(bboxes)
    img_out = draw_bboxes_and_keypoints(img_in, bboxes, keypoints_all=None)
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

        bbox = bboxes[max_idx]
        cropped_img = crop_and_resize(img_in, bbox, output_size)

        if write_images:
            if not cv2.imwrite(out_p, cropped_img):
                raise RuntimeError("Could not write image")
        # cv2.imshow("Cropped Image", cropped_img)
        # cv2.waitKey(0)

    else:
        print("Did not detect any faces")


if __name__ == "__main__":
    CLI(main, as_positional=False)
