import cv2
import matplotlib.pyplot as plt
import numpy as np
from jsonargparse import CLI
from tqdm import tqdm

from src.ssd import ssd_detect
from src.utils import draw_bboxes_and_keypoints

func_dict = {
    "ssd": ssd_detect,
}


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


def main(
    img_p: str = "data/ColorFERET-00472_940519_hr_small.png",
    out_p: str = "data/ColorFERET-00472_940519_hr_small_cropped.png",
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
