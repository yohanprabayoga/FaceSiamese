import numpy as np
import os
import cv2
from tensorflow.keras.optimizers import Adam
from siamese_network import build_model
import dlib

def preprocess_image(image_path):
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor('/home/pi/FaceSiamese/shape_predictor_68_face_landmarks.dat')
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to read image {image_path}")
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    if len(dets) == 0:
        print(f"No face detected in image {image_path}")
        return None
    shape = sp(gray, dets[0])
    face_chip = dlib.get_face_chip(img, shape, size=105)
    return face_chip

def create_pairs(dataset_path):
    pairs = []
    labels = []
    people = os.listdir(dataset_path)
    
    # Creating positive pairs
    for person in people:
        person_path = os.path.join(dataset_path, person)
        images = [os.path.join(person_path, img) for img in os.listdir(person_path)]
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                pairs.append([images[i], images[j]])
                labels.append(1)  # Positive pair

    # Creating negative pairs
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            person1_path = os.path.join(dataset_path, people[i])
            person2_path = os.path.join(dataset_path, people[j])
            img1 = os.path.join(person1_path, os.listdir(person1_path)[0])
            img2 = os.path.join(person2_path, os.listdir(person2_path)[0])
            pairs.append([img1, img2])
            labels.append(0)  # Negative pair

    return np.array(pairs), np.array(labels)

def preprocess_pairs(pairs):
    preprocessed_pairs = []
    valid_labels = []
    for pair in pairs:
        img1 = preprocess_image(pair[0])
        img2 = preprocess_image(pair[1])
        if img1 is not None and img2 is not None:
            preprocessed_pairs.append([img1, img2])
            valid_labels.append(1 if pair[0].split('/')[-2] == pair[1].split('/')[-2] else 0)
    return np.array(preprocessed_pairs), np.array(valid_labels)

if __name__ == "__main__":
    print("Creating pairs...")
    dataset_path = '/home/pi/FaceSiamese/dataset'
    pairs, labels = create_pairs(dataset_path)
    print(f"Total pairs created: {len(pairs)}")

    print("Preprocessing pairs...")
    pairs, labels = preprocess_pairs(pairs)
    print(f"Total pairs after preprocessing: {len(pairs)}")

    if len(pairs) == 0:
        print("No valid image pairs found. Please check the dataset and preprocessing.")
        exit()

    input_shape = (105, 105, 3)
    model = build_model(input_shape)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    pairs_reshaped = [np.array([pair[0] for pair in pairs]).reshape(-1, 105, 105, 3),
                      np.array([pair[1] for pair in pairs]).reshape(-1, 105, 105, 3)]

    print("Training the model...")
    model.fit(pairs_reshaped, labels, batch_size=32, epochs=10)
    model.save('/home/pi/FaceSiamese/models/siamese_model.h5')
    print("Model training completed and saved as siamese_model.h5.")
