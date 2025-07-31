import cv2
import os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()  # Create LBPH face recognizer
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # Load Haar cascade for face detection

def getImagesAndLabels(path):
    faceSamples = []
    Ids = []
    
    if not os.path.exists(path):
        print(f"Error: Directory '{path}' not found.")
        return [], []

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')  # Convert image to grayscale
            imageNp = np.array(pilImage, 'uint8')  # Convert image to numpy array
            Id = int(os.path.split(imagePath)[-1].split(".")[1])  # Extract ID from image filename
            faces = detector.detectMultiScale(imageNp)
            
            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y+h, x:x+w])  # Add face region to the list
                Ids.append(Id)  # Add the corresponding ID
        except Exception as e:
            print(f"Error processing image '{imagePath}': {e}")

    return faceSamples, Ids

faces, Ids = getImagesAndLabels('TrainingImage')
if faces and Ids:
    recognizer.train(faces, np.array(Ids))  # Train the recognizer with the collected faces
    recognizer.save('TrainingImageLabel/trainner.yml')  # Save the trained model
    print("Training completed successfully.")
else:
    print("No face images found for training.")
