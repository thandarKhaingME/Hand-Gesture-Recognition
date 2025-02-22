import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from pynput.keyboard import Key, Controller
import time
import pickle


class GameGestureController:
    def __init__(self, model_path='Hand_Gesture_Recognition.h5', label_encoder_path='label_encoder.pkl'):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load the label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Initialize keyboard controller
        self.keyboard = Controller()
        
        # Define gesture to key mappings for Asphalt Legends
        self.gesture_controls = {
            'fist': Key.space,        # Nitro boost
            'palm': 'w',              # Accelerate
            'peace': 'a',             # Turn left
            'ok': 'd',                # Turn right
            'like': 'x',              # Use item/power-up
            'dislike': 's',           # Brake/Reverse
            'stop': Key.esc,          # Pause game
            'one': Key.enter,         # Select/Confirm
            'two_up': Key.up,         # Navigate up
            'three': Key.down,        # Navigate down
            'four': 'e',              # Special action
            'mute': 'm'               # Mute game
        }
        
        # Gesture smoothing
        self.prev_gesture = None
        self.gesture_counter = 0
        self.min_gesture_frames = 3
        
        # Active keys tracking
        self.active_keys = set()
        
        # Debug info
        print("Available gestures:", self.label_encoder.classes_)

    def preprocess_landmarks(self, hand_landmarks):
        landmarks_flat = []
        for landmark in hand_landmarks.landmark:
            landmarks_flat.extend([landmark.x, landmark.y, landmark.z])
        return np.array([landmarks_flat])

    def release_all_keys(self):
        """Release all currently pressed keys"""
        for key in self.active_keys:
            self.keyboard.release(key)
        self.active_keys.clear()

    def execute_gesture_action(self, gesture, confidence):
        """Execute keyboard action based on detected gesture"""
        if gesture in self.gesture_controls and confidence > 0.7:  # Added confidence threshold
            key = self.gesture_controls[gesture]
            
            # Release previously pressed keys if different gesture
            if gesture != self.prev_gesture:
                self.release_all_keys()
                
                # Press the new key
                self.keyboard.press(key)
                self.active_keys.add(key)
                self.prev_gesture = gesture
        else:
            # Release all keys if no recognized gesture or low confidence
            self.release_all_keys()
            self.prev_gesture = None

    def run(self):
        """Main loop for gesture control"""
        cap = cv2.VideoCapture(0)
        
        try:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Failed to capture frame")
                    continue

                # Convert image and process
                image = cv2.flip(image, 1)  # Mirror image
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    # Get the first detected hand
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Preprocess landmarks and predict gesture
                    landmarks_processed = self.preprocess_landmarks(hand_landmarks)
                    prediction = self.model.predict(landmarks_processed, verbose=0)
                    predicted_idx = np.argmax(prediction[0])
                    confidence = prediction[0][predicted_idx]
                    
                    # Get gesture label using label encoder
                    predicted_gesture = self.label_encoder.inverse_transform([predicted_idx])[0]
                    
                    # Execute the gesture action
                    self.execute_gesture_action(predicted_gesture, confidence)
                    
                    # Display gesture and confidence on screen
                    cv2.putText(image, f"Gesture: {predicted_gesture}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (0, 255, 0), 2)
                    cv2.putText(image, f"Confidence: {confidence:.2f}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (0, 255, 0), 2)
                else:
                    # Release all keys if no hand detected
                    self.release_all_keys()

                # Display the frame
                cv2.imshow('Game Gesture Controller', image)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # Cleanup
            self.release_all_keys()
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()

if __name__ == "__main__":
    controller = GameGestureController()
    controller.run()