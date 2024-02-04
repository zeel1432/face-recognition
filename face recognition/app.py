import dlib
import os
import cv2

# Load the trained model
model = dlib.face_recognition_model_v1()
model.load('/lfw-funneled.tgz')

# Load the image to be recognized
image = cv2.imread('/2IMG_6823~2.jpg')

# Detect faces in the image
detector = dlib.get_frontal_face_detector()
faces = detector(image)

# Iterate over the detected faces
for face in faces:
    # Compute the face descriptor
    shape = model.get_shape(image, face)
    descriptor = model.compute_face_descriptor(image, shape)

    # Perform face recognition
    # Compare the descriptor with the descriptors of known faces in the dataset
    # Determine the identity of the recognized face

    # Display the result
    cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

# Show the image with recognized faces
cv2.imshow('Face Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
