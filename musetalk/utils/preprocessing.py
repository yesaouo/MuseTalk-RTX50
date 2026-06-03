import sys
from os import listdir, path
import subprocess
import numpy as np
import cv2
import pickle
import os
import json
import torch
from tqdm import tqdm

from face_alignment import FaceAlignment, LandmarksType


# MuseTalk placeholder if face detection fails
coord_placeholder = (0.0, 0.0, 0.0, 0.0)


# -----------------------------
# Initialize face-alignment
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    landmark_type = LandmarksType.TWO_D
except AttributeError:
    landmark_type = LandmarksType._2D

fa = FaceAlignment(
    landmark_type,
    flip_input=False,
    device=device,
    compile=False
)




def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized


def read_imgs(img_list):
    frames = []
    print("reading images...")
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def _mediapipe_478_landmarks(frame):
    return None

def _face_alignment_68_landmarks(frame):
    """
    Return face-alignment 68 landmarks in pixel coordinates.
    Input frame is BGR OpenCV image.
    """
    if frame is None:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    preds = fa.get_landmarks_from_image(rgb)

    if preds is None or len(preds) == 0:
        return None

    return preds[0].astype(np.float32)


def _get_face_landmarks(frame):
    """
    Replacement for the old mmpose wholebody face landmarks.

    Strategy:
    1. Use face-alignment to get 68 facial landmarks.
    2. Use MediaPipe 478-point mesh to overwrite MuseTalk's nose bridge indices:
       face_lm[28] = pts_478[6]
       face_lm[29] = pts_478[197]
       face_lm[30] = pts_478[195]
    3. If face-alignment fails but MediaPipe works, fallback to MediaPipe landmarks.
    """
    face_lm = _face_alignment_68_landmarks(frame)
    pts_478 = _mediapipe_478_landmarks(frame)

    if face_lm is None and pts_478 is None:
        return None

    if face_lm is None:
        face_lm = pts_478.copy()

    if pts_478 is not None and len(pts_478) > 197 and len(face_lm) > 30:
        face_lm[28] = pts_478[6]
        face_lm[29] = pts_478[197]
        face_lm[30] = pts_478[195]

    return face_lm.astype(np.int32)


def _safe_bbox_from_landmarks(face_land_mark, frame):
    """
    Create a safe bbox from landmarks.
    """
    h, w = frame.shape[:2]

    x1 = int(np.min(face_land_mark[:, 0]))
    y1 = int(np.min(face_land_mark[:, 1]))
    x2 = int(np.max(face_land_mark[:, 0]))
    y2 = int(np.max(face_land_mark[:, 1]))

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return coord_placeholder

    return (x1, y1, x2, y2)


def get_bbox_range(img_list, upperbondrange=0):
    frames = read_imgs(img_list)

    coords_list = []
    landmarks = []

    if upperbondrange != 0:
        print("get key_landmark and face bounding boxes with the bbox_shift:", upperbondrange)
    else:
        print("get key_landmark and face bounding boxes with the default value")

    average_range_minus = []
    average_range_plus = []

    for frame in tqdm(frames):
        face_land_mark = _get_face_landmarks(frame)

        if face_land_mark is None or len(face_land_mark) <= 30:
            coords_list += [coord_placeholder]
            continue

        half_face_coord = face_land_mark[29].copy()

        range_minus = int((face_land_mark[30] - face_land_mark[29])[1])
        range_plus = int((face_land_mark[29] - face_land_mark[28])[1])

        average_range_minus.append(range_minus)
        average_range_plus.append(range_plus)

        if upperbondrange != 0:
            half_face_coord[1] = upperbondrange + half_face_coord[1]

    if len(average_range_minus) == 0 or len(average_range_plus) == 0:
        return "No face detected. Cannot calculate bbox_shift range."

    text_range = (
        f"Total frame:「{len(frames)}」 "
        f"Manually adjust range : "
        f"[ -{int(sum(average_range_minus) / len(average_range_minus))}"
        f"~{int(sum(average_range_plus) / len(average_range_plus))} ] , "
        f"the current value: {upperbondrange}"
    )

    return text_range


def get_landmark_and_bbox(img_list, upperbondrange=0):
    frames = read_imgs(img_list)

    coords_list = []
    landmarks = []

    if upperbondrange != 0:
        print("get key_landmark and face bounding boxes with the bbox_shift:", upperbondrange)
    else:
        print("get key_landmark and face bounding boxes with the default value")

    average_range_minus = []
    average_range_plus = []

    for frame in tqdm(frames):
        face_land_mark = _get_face_landmarks(frame)

        if face_land_mark is None or len(face_land_mark) <= 30:
            coords_list += [coord_placeholder]
            continue

        half_face_coord = face_land_mark[29].copy()

        range_minus = int((face_land_mark[30] - face_land_mark[29])[1])
        range_plus = int((face_land_mark[29] - face_land_mark[28])[1])

        average_range_minus.append(range_minus)
        average_range_plus.append(range_plus)

        if upperbondrange != 0:
            # + means move down, - means move up
            half_face_coord[1] = upperbondrange + half_face_coord[1]

        half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]

        min_upper_bond = 0
        upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)

        f_landmark = (
            int(np.min(face_land_mark[:, 0])),
            int(upper_bond),
            int(np.max(face_land_mark[:, 0])),
            int(np.max(face_land_mark[:, 1]))
        )

        x1, y1, x2, y2 = f_landmark

        h, w = frame.shape[:2]

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if y2 - y1 <= 0 or x2 - x1 <= 0:
            fallback_bbox = _safe_bbox_from_landmarks(face_land_mark, frame)
            coords_list += [fallback_bbox]
            print("error bbox:", fallback_bbox)
        else:
            coords_list += [(x1, y1, x2, y2)]

    print("********************************************bbox_shift parameter adjustment**********************************************************")

    if len(average_range_minus) > 0 and len(average_range_plus) > 0:
        print(
            f"Total frame:「{len(frames)}」 "
            f"Manually adjust range : "
            f"[ -{int(sum(average_range_minus) / len(average_range_minus))}"
            f"~{int(sum(average_range_plus) / len(average_range_plus))} ] , "
            f"the current value: {upperbondrange}"
        )
    else:
        print("No valid face landmarks were detected.")

    print("*************************************************************************************************************************************")

    return coords_list, frames


if __name__ == "__main__":
    img_list = [
        "./results/lyria/00000.png",
        "./results/lyria/00001.png",
        "./results/lyria/00002.png",
        "./results/lyria/00003.png"
    ]

    crop_coord_path = "./coord_face.pkl"
    coords_list, full_frames = get_landmark_and_bbox(img_list)

    with open(crop_coord_path, "wb") as f:
        pickle.dump(coords_list, f)

    for bbox, frame in zip(coords_list, full_frames):
        if bbox == coord_placeholder:
            continue

        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        print("Cropped shape", crop_frame.shape)

    print(coords_list)