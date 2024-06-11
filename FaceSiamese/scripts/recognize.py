import cv2
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocess_image
from siamese_network import euclidean_distance
import os

def load_and_preprocess_image(image_path):
    print(f"Preprocessing image: {image_path}")
    image = preprocess_image(image_path)
    if image is None:
        print(f"Error: Failed to preprocess image {image_path}")
        return None
    return image.reshape(1, 105, 105, 3)

if __name__ == "__main__":
    print("Loading the model...")
    model = load_model('/home/pi/FaceSiamese/models/siamese_model.h5', custom_objects={'euclidean_distance': euclidean_distance})
    print("Model loaded successfully.")

    dataset_dir = '/home/pi/FaceSiamese/dataset/'
    new_images_dir = '/home/pi/FaceSiamese/new_images/'

    if not os.path.exists(new_images_dir):
        print(f"Error: New images directory not found at {new_images_dir}")
        exit()

    image_extensions = ('.jpg', '.jpeg', '.png')
    for new_image_filename in os.listdir(new_images_dir):
        if new_image_filename.lower().endswith(image_extensions):
            new_image_path = os.path.join(new_images_dir, new_image_filename)
            new_image = load_and_preprocess_image(new_image_path)
            if new_image is None:
                continue

            best_match = None
            best_distance = float('inf')
            threshold = 5

            for person_name in os.listdir(dataset_dir):
                person_dir = os.path.join(dataset_dir, person_name)
                for filename in os.listdir(person_dir):
                    if filename.lower().endswith(image_extensions):
                        known_image_path = os.path.join(person_dir, filename)
                        known_image = load_and_preprocess_image(known_image_path)
                        if known_image is None:
                            continue

                        distance = model.predict([known_image, new_image])[0][0]
                        print(f"Distance from {filename}: {distance}")
                        if distance < best_distance:
                            best_distance = distance
                            best_match = person_name

            if best_match and best_distance < threshold:
                print(f"Best match: {best_match} with distance {best_distance}")
                print(f"Recognized as {best_match}")
            else:
                print("No match found or different person")
