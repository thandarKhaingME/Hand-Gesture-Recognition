This project implements a Hand Gesture Recognition (HGR) system using MediaPipe for hand tracking, OpenCV for real-time video processing, and a Feedforward Neural Network (FNN) for gesture classification. 
The system recognizes hand gestures in real-time and can be used for applications like gaming control, virtual reality, or assistive technology.

Technologies Used
MediaPipe: For hand tracking and landmark detection.
OpenCV: For video capture, image processing, and visualization.
TensorFlow/Keras: For building and training the FNN model.
Python: The programming language used for the project.

Installation
Prerequisites
Python 3.7 or higher.(If you are using different python version, 3.8-3.10 is recommended not to have confusion with mediapipe.)
A webcam for real-time gesture recognition.

Note
Please check the requirement.txt file to downlaod all the necessary libraies.(pip install -r requirements.txt)
Also make sure you have a webcam and no other app is using at the moment then run controller.py, (python controller.py,click play button or ctrl f5). 

How It Works
Hand Tracking: MediaPipe detects hand landmarks from the webcam feed.
Feature Extraction: The landmarks are preprocessed and fed into the FNN model.
Gesture Classification: The FNN model predicts the gesture based on the input features.
Output: The recognized gesture is displayed on the screen.

Customization
Train Your Own Model: Use the train_model.py script to train a new FNN model on your dataset.
Add New Gestures: Update the label encoder and retrain the model to recognize additional gestures.

