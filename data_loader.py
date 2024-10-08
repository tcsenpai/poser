import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from colorama import Fore, init

# Initialize colorama
init(autoreset=True)

def load_datasets(dataset_path, img_size=(224, 224)):
    good_path = os.path.join(dataset_path, 'good')
    bad_path = os.path.join(dataset_path, 'bad')

    data = []
    labels = []

    # Load good posture images
    good_images = os.listdir(good_path)
    for img_name in good_images:
        img_path = os.path.join(good_path, img_name)
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img)
        data.append(img_array)
        labels.append(1)

    # Load bad posture images
    bad_images = os.listdir(bad_path)
    for img_name in bad_images:
        img_path = os.path.join(bad_path, img_name)
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img)
        data.append(img_array)
        labels.append(0)

    # Convert lists to numpy arrays
    X = np.array(data)
    y = np.array(labels)

    # Normalize pixel values to be between 0 and 1
    X = X.astype('float32') / 255.0

    print(Fore.CYAN + f"Total number of samples: {len(X)}")
    print(Fore.CYAN + f"Number of good posture samples: {len(good_images)}")
    print(Fore.CYAN + f"Number of bad posture samples: {len(bad_images)}")
    print(Fore.CYAN + f"Shape of X: {X.shape}")
    print(Fore.CYAN + f"Shape of y: {y.shape}")

    # Split the data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(X, y, test_size=0.2, random_state=42)

    print(Fore.CYAN + f"Shape of train_data: {train_data.shape}")
    print(Fore.CYAN + f"Shape of train_labels: {train_labels.shape}")
    print(Fore.CYAN + f"Shape of val_data: {val_data.shape}")
    print(Fore.CYAN + f"Shape of val_labels: {val_labels.shape}")

    return train_data, train_labels, val_data, val_labels