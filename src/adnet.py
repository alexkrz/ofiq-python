import cv2
import numpy as np
import onnxruntime as ort


def draw_landmarks(img: np.ndarray, landmarks: np.ndarray):
    for idx in range(len(landmarks)):
        x, y = landmarks[idx]
        cv2.circle(img, (x, y), 3, (255, 0, 0), cv2.FILLED)
        cv2.putText(img, str(idx), (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.imshow("Landmarks", img)
    cv2.waitKey(0)


def compute_landmarks(
    img: np.ndarray,
    model_p: str,
) -> np.ndarray:
    # Load model
    # model = cv2.dnn.readNetFromONNX(model_p)  # Cannot read ADNet model with OpenCV
    ort_sess = ort.InferenceSession(model_p)
    input_size = (256, 256)

    # Set up data
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
    outputs = ort_sess.run(["output"], {"input": blob})
    landmarks: np.ndarray = outputs[0]
    landmarks = landmarks.squeeze()  # Squeeze out batch dimension
    # print(landmarks.shape)
    # De-normalize landmarks
    landmarks = 127.5 * (landmarks + 1)
    # Transform x-coordinates
    landmarks[:, 0] = landmarks[:, 0] * (W / input_size[0])
    # Transform y-coordinates
    landmarks[:, 1] = landmarks[:, 1] * (H / input_size[1])
    landmarks = landmarks.astype("int")
    return landmarks
