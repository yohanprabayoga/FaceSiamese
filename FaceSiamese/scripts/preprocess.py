import cv2
import dlib
import os

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('/home/pi/FaceSiamese/shape_predictor_68_face_landmarks.dat')

def preprocess_image(image_path):
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

def preprocess_dataset(dataset_path):
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            face = preprocess_image(img_path)
            if face is not None:
                cv2.imwrite(img_path, face)
                print(f"Processed and saved: {img_path}")
            else:
                print(f"Failed to process: {img_path}")

if __name__ == "__main__":
    print("Starting dataset preprocessing...")
    preprocess_dataset('/home/pi/FaceSiamese/dataset')
    print("Dataset preprocessing completed.")
