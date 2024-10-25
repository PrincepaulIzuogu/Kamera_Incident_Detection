import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

def load_data(data_dir):
    """Loads images from the data directory and labels them as fall or no fall."""
    images = []
    labels = []

    for class_label in ['fall', 'no_fall']:
        subdir_path = os.path.join(data_dir, class_label)
        label = 1 if class_label == 'fall' else 0  # Label fall frames as 1, no fall as 0

        for video_folder in os.listdir(subdir_path):
            video_path = os.path.join(subdir_path, video_folder)
            if not os.path.isdir(video_path):
                continue

            for img_file in os.listdir(video_path):
                img_path = os.path.join(video_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue  # Skip if the image is not readable
                img = cv2.resize(img, (64, 64))  # Resize image to 64x64 for faster training
                images.append(img)
                labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

def preprocess_data(images, labels):
    """Preprocesses the images and splits the data into training and test sets."""
    images = images / 255.0  # Normalize pixel values
    labels = to_categorical(labels, 2)  # Convert labels to one-hot encoding

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def build_model():
    """Builds a simple CNN model for fall detection."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')  # Output layer for two classes (fall, no fall)
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    data_dir = "../data/frames"  # Path to the directory where frames are stored
    images, labels = load_data(data_dir)

    print(f"Total images: {len(images)}")
    print(f"Total labels: {len(labels)}")

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(images, labels)

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Build and train the model
    model = build_model()
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Save the model
    model.save("../models/fall_detection_model.h5")
    print("Model training complete and saved!")
