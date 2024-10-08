import cv2
import numpy as np

def preprocess_image(frame):
    # Resize and normalize the image
    img = cv2.resize(frame, (224, 224))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def detect_posture(frame, model):
    preprocessed = preprocess_image(frame)
    prediction = model.predict(preprocessed)[0][0]
    
    if prediction > 0.5:
        return "Good"
    else:
        return "Bad"