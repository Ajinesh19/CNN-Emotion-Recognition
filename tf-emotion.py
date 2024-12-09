import cv2
import tensorflow as tf
import numpy as np

# Load the trained model (ensure your model is saved first)
model = tf.keras.models.load_model('emotion_model.h5')

# Emotion labels (change according to your dataset)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using OpenCV's Haar cascades (make sure you have the classifier XML file)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop the face region
        face = frame[y:y+h, x:x+w]
        
        # Resize the face to match the input size of the model
        face_resized = cv2.resize(face, (48, 48))

        # Preprocess the face image
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)  # Convert to RGB
        face_resized = np.expand_dims(face_resized, axis=0)  # Add batch dimension
        face_resized = face_resized / 255.0  # Normalize the image

        # Predict emotion
        predictions = model.predict(face_resized)
        emotion_index = np.argmax(predictions[0])
        emotion = emotion_labels[emotion_index]

        # Draw a rectangle around the face and display the emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

itha athula podu