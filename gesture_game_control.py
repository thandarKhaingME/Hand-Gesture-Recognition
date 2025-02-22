import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import joblib
import time

# Load trained model
model = joblib.load("gesture_model.ipynb")  

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define gesture-to-key mappings (modify as needed for your game)
gesture_to_key = {
    "fist": "space",       # Jump (Mario)
    "palm": "w",           # Move forward
    "peace": "d",          # Move right
    "ok": "a",             # Move left
    "like": "up",          # Accelerate (Asphalt)
    "dislike": "down",     # Brake
    "stop": "s",           # Stop character
    "one": "right",        # Move right
    "two_up": "left",      # Move left
    "three": "enter",      # Start/select
    "four": "esc",         # Exit menu
    "mute": "m"           # Mute game
}

# Function to process frame and detect hand gestures
def detect_gesture(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks as a flattened array
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            
            if landmarks.shape[0] == 63:  # Ensure correct shape
                # Predict gesture using trained FNN model
                predicted_label = model.predict([landmarks])[0]
                
                # Perform action based on gesture
                if predicted_label in gesture_to_key:
                    key = gesture_to_key[predicted_label]
                    print(f"Detected Gesture: {predicted_label} → Pressing Key: {key}")
                    
                    # Simulate key press
                    pyautogui.press(key)
                
            # Draw landmarks on hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    return frame

# Open webcam for real-time detection
cap = cv2.VideoCapture(0)

print("Starting Hand Gesture Control... Open the game and use gestures to play!")
time.sleep(3)  # Wait before starting

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_gesture(frame)

    # Show webcam feed
    cv2.imshow("Hand Gesture Control", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
