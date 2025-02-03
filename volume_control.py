# # import cv2
# # import mediapipe as mp
# # import pyautoqui
# #
# # webcam = cv2.VideoCapture(0)
# # while True:
# #     _ , image = webcam.read()
# #     cv2.imshow("Hand Volume Control using python",image)
# #     cv2.waitkey(10)

# new code
#
# import cv2
# import mediapipe as mp
# import numpy as np
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# from comtypes import CLSCTX_ALL
# import math
#
# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
#
# # Initialize Pycaw for system volume control
# devices = AudioUtilities.GetSpeakers()
# interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# volume = interface.QueryInterface(IAudioEndpointVolume)
# vol_range = volume.GetVolumeRange()  # (-65.25, 0.0)
# min_vol, max_vol = vol_range[0], vol_range[1]
#
# # Variable to store last volume level
# last_volume_level = volume.GetMasterVolumeLevel()
#
# # Open webcam
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Convert frame to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Process frame and detect hands
#     results = hands.process(rgb_frame)
#
#     # Check if hands are detected
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#             # Get landmark positions of thumb (4) and index finger (8)
#             thumb = hand_landmarks.landmark[4]
#             index_finger = hand_landmarks.landmark[8]
#
#             # Convert normalized coordinates to pixel values
#             h, w, _ = frame.shape
#             x1, y1 = int(thumb.x * w), int(thumb.y * h)
#             x2, y2 = int(index_finger.x * w), int(index_finger.y * h)
#
#             # Draw circles on thumb and index finger
#             cv2.circle(frame, (x1, y1), 10, (255, 0, 0), -1)
#             cv2.circle(frame, (x2, y2), 10, (255, 0, 0), -1)
#             cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
#
#             # Calculate distance between thumb and index finger
#             distance = math.hypot(x2 - x1, y2 - y1)
#
#             # Convert distance to volume level
#             volume_level = np.interp(distance, [30, 200], [min_vol, max_vol])
#             volume.SetMasterVolumeLevel(volume_level, None)
#             last_volume_level = volume_level  # Store last volume level
#
#             # Display volume level
#             vol_percent = np.interp(distance, [30, 200], [0, 100])
#             cv2.putText(frame, f'Volume: {int(vol_percent)}%', (50, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
#     else:
#         # If no hands detected, maintain last volume level
#         volume.SetMasterVolumeLevel(last_volume_level, None)
#
#     # Show output
#     cv2.imshow("Hand Gesture Volume Control", frame)
#
#     # Exit on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Cleanup
# cap.release()
# cv2.destroyAllWindows()

#new code 2

# import cv2
# import mediapipe as mp
# import numpy as np
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# from comtypes import CLSCTX_ALL
# import math
# import collections
#
# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
#
# # Initialize Pycaw for system volume control
# devices = AudioUtilities.GetSpeakers()
# interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# volume = interface.QueryInterface(IAudioEndpointVolume)
# vol_range = volume.GetVolumeRange()  # (-65.25, 0.0)
# min_vol, max_vol = vol_range[0], vol_range[1]
#
# # Variable to store last volume level
# last_volume_level = volume.GetMasterVolumeLevel()
#
# # Circular motion detection variables
# index_finger_positions = collections.deque(maxlen=30)  # Store last 30 index finger positions
#
# # Open webcam
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Convert frame to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Process frame and detect hands
#     results = hands.process(rgb_frame)
#
#     # Check if hands are detected
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#             # Get landmark positions of thumb (4) and index finger (8)
#             thumb = hand_landmarks.landmark[4]
#             index_finger = hand_landmarks.landmark[8]
#
#             # Convert normalized coordinates to pixel values
#             h, w, _ = frame.shape
#             x1, y1 = int(thumb.x * w), int(thumb.y * h)
#             x2, y2 = int(index_finger.x * w), int(index_finger.y * h)
#
#             # Draw circles on thumb and index finger
#             cv2.circle(frame, (x1, y1), 10, (255, 0, 0), -1)
#             cv2.circle(frame, (x2, y2), 10, (255, 0, 0), -1)
#             cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
#
#             # Calculate distance between thumb and index finger
#             distance = math.hypot(x2 - x1, y2 - y1)
#
#             # Convert distance to volume level
#             volume_level = np.interp(distance, [30, 200], [min_vol, max_vol])
#             volume.SetMasterVolumeLevel(volume_level, None)
#             last_volume_level = volume_level  # Store last volume level
#
#             # Display volume level
#             vol_percent = np.interp(distance, [30, 200], [0, 100])
#             cv2.putText(frame, f'Volume: {int(vol_percent)}%', (50, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
#
#             # Track index finger movement
#             index_finger_positions.append((x2, y2))
#
#             # Detect circular motion based on past movements
#             if len(index_finger_positions) >= 30:
#                 x_positions = [pos[0] for pos in index_finger_positions]
#                 y_positions = [pos[1] for pos in index_finger_positions]
#
#                 x_movement = max(x_positions) - min(x_positions)
#                 y_movement = max(y_positions) - min(y_positions)
#
#                 if x_movement > 50 and y_movement > 50:  # If movement is large in both directions
#                     print("🛑 Circular motion detected! Closing camera...")
#                     cap.release()
#                     cv2.destroyAllWindows()
#                     exit()
#
#     else:
#         # If no hands detected, maintain last volume level
#         volume.SetMasterVolumeLevel(last_volume_level, None)
#
#     # Show output
#     cv2.imshow("Hand Gesture Volume Control", frame)
#
#     # Exit on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Cleanup
# cap.release()
# cv2.destroyAllWindows()

# new code 3
# import cv2
# import mediapipe as mp
# import numpy as np
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# from comtypes import CLSCTX_ALL
# import math
#
# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
#
# # Initialize Pycaw for system volume control
# devices = AudioUtilities.GetSpeakers()
# interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# volume = interface.QueryInterface(IAudioEndpointVolume)
# vol_range = volume.GetVolumeRange()  # (-65.25, 0.0)
# min_vol, max_vol = vol_range[0], vol_range[1]
#
# # Variable to store last volume level
# last_volume_level = volume.GetMasterVolumeLevel()
#
# # Open webcam
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Convert frame to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Process frame and detect hands
#     results = hands.process(rgb_frame)
#
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#             # Get landmark positions of thumb (4) and index finger (8)
#             thumb = hand_landmarks.landmark[4]
#             index_finger = hand_landmarks.landmark[8]
#
#             # Convert normalized coordinates to pixel values
#             h, w, _ = frame.shape
#             x1, y1 = int(thumb.x * w), int(thumb.y * h)
#             x2, y2 = int(index_finger.x * w), int(index_finger.y * h)
#
#             # Draw circles on thumb and index finger
#             cv2.circle(frame, (x1, y1), 10, (255, 0, 0), -1)
#             cv2.circle(frame, (x2, y2), 10, (255, 0, 0), -1)
#             cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
#
#             # Calculate distance between thumb and index finger
#             distance = math.hypot(x2 - x1, y2 - y1)
#
#             # Convert distance to volume level
#             volume_level = np.interp(distance, [30, 200], [min_vol, max_vol])
#             volume.SetMasterVolumeLevel(volume_level, None)
#             last_volume_level = volume_level  # Store last volume level
#
#             # Display volume level
#             vol_percent = np.interp(distance, [30, 200], [0, 100])
#             cv2.putText(frame, f'Volume: {int(vol_percent)}%', (50, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
#
#     else:
#         # If no hands detected, maintain last volume level
#         volume.SetMasterVolumeLevel(last_volume_level, None)
#
#     # Show output
#     cv2.imshow("Hand Gesture Volume Control", frame)
#
#     # Exit on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Cleanup
# cap.release()
# cv2.destroyAllWindows()

# new code 4
import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Initialize Pycaw for system volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
vol_range = volume.GetVolumeRange()  # (-65.25, 0.0)
min_vol, max_vol = vol_range[0], vol_range[1]

# Variable to store last volume level
last_volume_level = volume.GetMasterVolumeLevel()

# Open webcam
cap = cv2.VideoCapture(0)

# Counter for fist detection
fist_counter = 0
fist_threshold = 10  # Number of consecutive frames to confirm a fist

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame and detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions of thumb (4) and index finger (8)
            thumb = hand_landmarks.landmark[4]
            index_finger = hand_landmarks.landmark[8]

            # Convert normalized coordinates to pixel values
            h, w, _ = frame.shape
            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index_finger.x * w), int(index_finger.y * h)

            # Draw circles on thumb and index finger
            cv2.circle(frame, (x1, y1), 10, (255, 0, 0), -1)
            cv2.circle(frame, (x2, y2), 10, (255, 0, 0), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Calculate distance between thumb and index finger
            distance = math.hypot(x2 - x1, y2 - y1)

            # Convert distance to volume level
            volume_level = np.interp(distance, [30, 200], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(volume_level, None)
            last_volume_level = volume_level  # Store last volume level

            # Display volume level
            vol_percent = np.interp(distance, [30, 200], [0, 100])
            cv2.putText(frame, f'Volume: {int(vol_percent)}%', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Improved fist detection logic
            fingertip_ids = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky
            knuckle_ids = [2, 6, 10, 14, 18]  # Middle joints of the fingers
            fingers_folded = 0  # Counter for folded fingers

            for tip, knuckle in zip(fingertip_ids, knuckle_ids):
                fingertip_y = hand_landmarks.landmark[tip].y
                knuckle_y = hand_landmarks.landmark[knuckle].y

                if fingertip_y > knuckle_y:  # If the fingertip is below the knuckle, the finger is folded
                    fingers_folded += 1

            if fingers_folded == 5:  # All fingers are folded (fist detected)
                fist_counter += 1
                cv2.putText(frame, "Fist Detected! Closing Camera...", (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                if fist_counter >= fist_threshold:
                    print("Fist detected consistently. Closing camera...")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()  # Exit program

            else:
                fist_counter = 0  # Reset counter if no fist detected

    else:
        # If no hands detected, maintain last volume level
        volume.SetMasterVolumeLevel(last_volume_level, None)

    # Show output
    cv2.imshow("Hand Gesture Volume Control", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()



