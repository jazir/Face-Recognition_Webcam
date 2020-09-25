import cv2
import numpy as np
import face_recognition
import os

KNOWN_FACES_DIR = "known_faces"
# UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.5
FRAME_THICKNESS = 3  # Thickness of box in pixels
FONT_THICKNESS = 2
MODEL = "cnn"  # 'hog' is another model

video = cv2.VideoCapture(0)  # Webcam
print("video taken-top")
print("loading the known faces")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

print("Checking unknown faces")

while True:
    # print(filename)
    # image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
    #video = cv2.VideoCapture(0)  # Webcam

    success, image = video.read()
    print("image taken")
    print(f"status:{success}")
    image = cv2.resize(image, (0, 0), None, 0.25, 0.25) # To resize the image (for speeding up)
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # No need to encode,
    # because we already use cv2 to capture. So, this will be taken care of

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found:{match}")
            #  Box for face
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [0, 255, 0]
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            #  Box for name
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            color = [0, 0, 255]
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
                200, 200, 200), FONT_THICKNESS)
        else:
            print("No match found")
    cv2.imshow(filename, image)
    if cv2.waitKey(1) &0xFF == ord("q"):
        break
        #cv2.waitKey(0)
        #cv2.destroyWindow(filename)
