import cv2
import os
import numpy as np
from PIL import Image

# Create a face recognizer object
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the pre-trained face detector model
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def getImagesAndLabels(path):
    """
    Reads images from the given directory, detects faces, and extracts labels from filenames.
    :param path: Path to the directory containing training images.
    :return: Tuple of face samples and their corresponding labels.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Training directory '{path}' not found.")

    # Get all image file paths in the directory
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.png'))]
    faceSamples = []
    Ids = []

    for imagePath in imagePaths:
        try:
            # Load image and convert it to grayscale
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')

            # Extract ID from filename (assuming format 'subject.ID.jpg' or 'subject.ID.png')
            Id = int(os.path.split(imagePath)[-1].split(".")[1])

            # Detect faces in the image
            faces = detector.detectMultiScale(imageNp)
            if len(faces) == 0:
                print(f"No faces detected in image: {imagePath}")
                continue

            # Add detected faces and their IDs
            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y+h, x:x+w])
                Ids.append(Id)
        except Exception as e:
            print(f"Error processing image {imagePath}: {e}")

    return faceSamples, Ids

def train_face_recognizer(training_path, output_path):
    """
    Trains the face recognizer using the images in the training directory.
    :param training_path: Path to the directory containing training images.
    :param output_path: Path to save the trained model.
    """
    print("Starting training process...")
    faces, Ids = getImagesAndLabels(training_path)

    if len(faces) == 0 or len(Ids) == 0:
        raise ValueError("No valid faces found in the training data. Ensure images contain detectable faces.")

    # Train the recognizer
    recognizer.train(faces, np.array(Ids))

    # Save the trained model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    recognizer.save(output_path)
    print("Training complete. Model saved to:", output_path)

if __name__ == "__main__":
    # Define paths
    training_path = 'TrainingImage'  # Directory containing training images
    output_path = 'TrainingImageLabel/trainner.yml'  # Path to save the trained model

    # Train and save the model
    try:
        train_face_recognizer(training_path, output_path)
    except Exception as e:
        print("Error:", e)
