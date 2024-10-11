import cv2
import numpy as np


def draw_landmarks(img_p: str, landmarks: np.ndarray):
    img = cv2.imread(img_p)
    for idx in range(len(landmarks)):
        x, y = landmarks[idx]
        cv2.circle(img, (x, y), 3, (255, 0, 0), cv2.FILLED)
        cv2.putText(img, str(idx), (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.imshow("Landmarks", img)
    cv2.waitKey(0)


def compute_landmarks(
    img_p: str,
    model_p: str,
) -> np.ndarray:
    # Load model
    model = cv2.dnn.readNetFromONNX(model_p)
    input_size = (112, 112)
    output_layer_names = model.getUnconnectedOutLayersNames()

    # Set up data
    img = cv2.imread(img_p)
    W, H, C = img.shape
    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=1 / (127.5),
        size=input_size,
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
        crop=False,
    )

    # Run forward pass
    model.setInput(blob)
    outputs = model.forward(output_layer_names)
    landmarks: np.ndarray = outputs[1]
    landmarks = landmarks.squeeze()  # Squeeze out batch dimension
    landmarks = landmarks.reshape(-1, 2)  # Convert landmarks to 2D array
    # print(landmarks.shape)
    # De-normalize landmarks
    landmarks = 127.5 * (landmarks)
    # Transform x-coordinates
    landmarks[:, 0] = landmarks[:, 0] * (W / input_size[0])
    # Transform y-coordinates
    landmarks[:, 1] = landmarks[:, 1] * (H / input_size[1])
    landmarks = landmarks.astype("int")
    return landmarks


if __name__ == "__main__":
    img_p = "data/c-07-twofaces_cropped.png"
    model_p = "checkpoints/pfld/pfld.onnx"
    landmarks = compute_landmarks(img_p, model_p)
    draw_landmarks(img_p, landmarks)
