import cv2
import numpy as np
import tensorflow as tf

def preprocess_image(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def detect_posture(frame, model):
    preprocessed = preprocess_image(frame)
    prediction = model(preprocessed, training=False)
    prediction = tf.nn.softmax(prediction.logits)
    
    if prediction[0][1] > 0.5:
        return "Good"
    else:
        return "Bad"