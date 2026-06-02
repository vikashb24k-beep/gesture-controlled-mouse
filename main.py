import math
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


CAMERA_INDEX = 0
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.7
SMOOTHING = 5
CLICK_DISTANCE = 55
SCROLL_DISTANCE = 40
SCROLL_AMOUNT = 30
MOVE_DEAD_ZONE = 2
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


def distance(point_a, point_b):
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def landmark_points(hand_landmarks, frame_width, frame_height):
    points = {}
    for index, landmark in enumerate(hand_landmarks):
        points[index] = (
            int(landmark.x * frame_width),
            int(landmark.y * frame_height),
        )
    return points


def draw_hand(frame, points):
    for start, end in HAND_CONNECTIONS:
        cv.line(frame, points[start], points[end], (0, 200, 255), 2, cv.LINE_AA)

    for point in points.values():
        cv.circle(frame, point, 4, (0, 255, 0), -1, cv.LINE_AA)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH.name}. Download the MediaPipe hand landmarker model first."
        )

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0

    screen_w, screen_h = pyautogui.size()
    prev_x = 0
    prev_y = 0
    clicked = False
    frame_number = 0

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=DETECTION_CONFIDENCE,
        min_hand_presence_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv.VideoCapture(CAMERA_INDEX, cv.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            "Camera could not be opened. Close other camera apps or change CAMERA_INDEX."
        )

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                print("Could not read from camera. Retrying...")
                if cv.waitKey(50) == 27:
                    break
                continue

            frame = cv.flip(frame, 1)
            frame_h, frame_w, _ = frame.shape
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_number += 1
            timestamp_ms = int(frame_number * 1000 / max(cap.get(cv.CAP_PROP_FPS), 30))
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]
                points = landmark_points(hand_landmarks, frame_w, frame_h)
                draw_hand(frame, points)

                thumb_tip = points[4]
                index_tip = points[8]
                middle_tip = points[12]
                pinky_tip = points[20]

                curr_x = np.interp(thumb_tip[0], [0, frame_w], [0, screen_w])
                curr_y = np.interp(thumb_tip[1], [0, frame_h], [0, screen_h])
                curr_x = prev_x + (curr_x - prev_x) / SMOOTHING
                curr_y = prev_y + (curr_y - prev_y) / SMOOTHING

                if abs(curr_x - prev_x) > MOVE_DEAD_ZONE or abs(curr_y - prev_y) > MOVE_DEAD_ZONE:
                    pyautogui.moveTo(curr_x, curr_y)

                prev_x = curr_x
                prev_y = curr_y

                if distance(thumb_tip, index_tip) < CLICK_DISTANCE:
                    if not clicked:
                        pyautogui.click()
                        clicked = True
                else:
                    clicked = False

                if distance(thumb_tip, middle_tip) < SCROLL_DISTANCE:
                    pyautogui.scroll(SCROLL_AMOUNT)
                elif distance(thumb_tip, pinky_tip) < SCROLL_DISTANCE:
                    pyautogui.scroll(-SCROLL_AMOUNT)

            cv.putText(
                frame,
                "ESC: quit | thumb: move | thumb+index: click | thumb+middle: up | thumb+pinky: down",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv.LINE_AA,
            )
            cv.imshow("Hand Gesture Mouse", frame)

            if cv.waitKey(1) == 27:
                break
    finally:
        landmarker.close()
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
