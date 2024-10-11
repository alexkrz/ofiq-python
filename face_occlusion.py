import cv2
import matplotlib.pyplot as plt
import numpy as np
from jsonargparse import CLI

from src.adnet import compute_landmarks


def get_face_mask(landmarks: np.ndarray, width: int, height: int, alpha: float):

    # Select eyes_midpoint, chin and contour landmarks
    left_eye_corners = np.stack([landmarks[60], landmarks[64]])
    right_eye_corners = np.stack([landmarks[68], landmarks[72]])

    eye_corners = np.concatenate([left_eye_corners, right_eye_corners])
    eyes_midpoint = np.sum(eye_corners, axis=0) / len(eye_corners)
    eyes_midpoint = eyes_midpoint.astype("int")
    eyes_midpoint = eyes_midpoint[None, :]  # Add extra dimension

    chin = landmarks[16]
    chin = chin[None, :]  # Add extra dimension
    contour_indices = [0, 7, 25, 32]
    contour_points = []
    for idx in contour_indices:
        contour_points.append(landmarks[idx])
    contour = np.array(contour_points)

    chin_midpoint_vector = eyes_midpoint - chin
    top_of_forehead = eyes_midpoint + alpha * chin_midpoint_vector

    # Fit ellipse to landmark points
    ellipse_points = np.concatenate([contour, chin, top_of_forehead])
    ellipse_points = np.array(ellipse_points, dtype=np.int32)
    fitted_ellipse = cv2.fitEllipse(ellipse_points)
    center = (int(fitted_ellipse[0][0]), int(fitted_ellipse[0][1]))  # Ellipse center (x, y)
    axes = (
        int(fitted_ellipse[1][0] / 2),
        int(fitted_ellipse[1][1] / 2),
    )  # Semi-major and semi-minor axes
    angle = int(fitted_ellipse[2])  # Rotation angle
    poly_points = cv2.ellipse2Poly(center, axes, angle, 0, 360, 10)

    # Discard ellipse points which are not on forehead
    poly_points_list = []
    chin_midpoint_vector = chin_midpoint_vector.squeeze()
    for p in poly_points:
        if np.dot(p - chin, chin_midpoint_vector) > 1.1 * np.dot(
            chin_midpoint_vector, chin_midpoint_vector
        ):
            poly_points_list.append(p)
    poly_points = np.array(poly_points_list, dtype=np.int32)

    # Add poly_points to landmark points
    landmarks = np.concatenate([landmarks, poly_points])
    landmarks = landmarks.astype("int")

    # Fit convex hull
    hull_points = cv2.convexHull(landmarks).squeeze()

    rect = cv2.boundingRect(hull_points)
    rect_x, rect_y, rect_width, rect_height = rect

    b = int(rect_y - rect_height * 0.05)
    d = int(rect_y + rect_height * 1.05)
    a = int(rect_x + rect_width / 2.0 - (d - b) / 2.0)
    c = int(rect_x + rect_width / 2.0 + (d - b) / 2.0)

    # Compute relative landmarks on cropped image
    img_size = 224
    hull_point_list = []
    for idx in range(len(hull_points)):
        point = hull_points[idx]
        point = (point - np.array([a, b])) / (d - b) * img_size
        hull_point_list.append(point)
    hull_points = np.array(hull_point_list, dtype=np.int32)

    # Generate mask from convex hull
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(hull_points, dtype=np.int32), 1)

    face_region = np.zeros((height, width), dtype=np.uint8)
    mask_rescaled = cv2.resize(mask, (c - a, d - b), interpolation=cv2.INTER_NEAREST)

    left, top, right, bottom = 0, 0, mask_rescaled.shape[0], mask_rescaled.shape[1]
    an, bn, cn, dn = a, b, c, d

    if a < 0:
        left -= a
        an = 0
    if c > width:
        right -= c - width
        cn = width
    if b < 0:
        top -= b
        bn = 0
    if d > height:
        bottom -= d - height
        dn = height

    crop = mask_rescaled[top:bottom, left:right]
    face_region[bn:dn, an:cn] = crop

    return face_region


def main(
    img_p: str = "data/c-07-twofaces_cropped.png",
    model_p: str = "checkpoints/adnet/adnet_ofiq.onnx",
):
    img = cv2.imread(img_p)
    width, height, channels = img.shape
    landmarks = compute_landmarks(img_p, model_p)
    face_mask = get_face_mask(landmarks, width, height, alpha=0.0)

    # Plot face_mask on img
    face_mask = cv2.cvtColor(face_mask, cv2.COLOR_GRAY2BGR)  # Extend face_mask channels
    face_mask[:, :, 2] = face_mask[:, :, 2] * 120.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blended_image = cv2.addWeighted(img, 0.7, face_mask, 0.3, 0.0)

    plt.imshow(blended_image)
    plt.show()


if __name__ == "__main__":
    CLI(main, as_positional=False)
