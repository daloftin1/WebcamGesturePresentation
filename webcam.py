
import cv2
import mediapipe as mp
import pyautogui
import time
from collections import deque

# MediaPipe hand tracking
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1
)

# Open MacBook webcam
camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not camera.isOpened():
    print("Error: Could not access webcam.")
    raise SystemExit(1)

# Gesture detection settings
positions = deque(maxlen=12)
last_swipe = 0
cooldown = 1.5
gesture = "Waiting for gesture..."

# Hand landmark connections
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

print("Webcam gesture detection started!")
print("Swipe right = Next slide")
print("Swipe left = Previous slide")
print("Press Q to quit.")

try:
    with HandLandmarker.create_from_options(options) as detector:
        while True:
            success, frame = camera.read()

            if not success:
                print("Camera frame failed.")
                break

            # Mirror webcam view
            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            # Convert image for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # Detect hand
            result = detector.detect(image)

            current_time = time.time()

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]

                # Draw hand landmarks
                for landmark in hand:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame, (x, y), 4,
                        (0, 255, 0), -1
                    )

                # Draw hand connections
                for start, end in connections:
                    x1 = int(hand[start].x * width)
                    y1 = int(hand[start].y * height)
                    x2 = int(hand[end].x * width)
                    y2 = int(hand[end].y * height)

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        2
                    )

                # Track palm position
                palm_x = hand[9].x
                positions.append(palm_x)

                if len(positions) == positions.maxlen:
                    movement = positions[-1] - positions[0]

                    if current_time - last_swipe > cooldown:

                        if movement > 0.22:
                            gesture = "SWIPE RIGHT - NEXT SLIDE"

                            # Send right arrow to PowerPoint
                            pyautogui.press("right")

                            print(gesture)
                            last_swipe = current_time
                            positions.clear()

                        elif movement < -0.22:
                            gesture = "SWIPE LEFT - PREVIOUS SLIDE"

                            # Send left arrow to PowerPoint
                            pyautogui.press("left")

                            print(gesture)
                            last_swipe = current_time
                            positions.clear()

            else:
                positions.clear()

            # Display gesture status
            cv2.putText(
                frame,
                gesture,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                "Webcam Gesture Presentations",
                frame
            )

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
