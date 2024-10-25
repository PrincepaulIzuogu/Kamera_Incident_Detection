import cv2
import numpy as np
from tensorflow.keras.models import load_model

def load_trained_model(model_path):
    """Load the pre-trained model."""
    return load_model(model_path)

def predict_frame(model, frame):
    """Make a prediction on a single frame."""
    img = cv2.resize(frame, (64, 64))
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match model input shape
    img = img / 255.0  # Normalize
    prediction = model.predict(img)
    return prediction

def real_time_fall_detection(model_path, video_source=0):
    """Perform real-time fall detection using a pre-trained model."""
    model = load_trained_model(model_path)

    # Use webcam (video_source=0) or video file (video_source='video.mp4')
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        prediction = predict_frame(model, frame)
        fall_prob = prediction[0][1]  # Probability of 'fall' class
        no_fall_prob = prediction[0][0]  # Probability of 'no fall' class

        # Determine the label based on higher probability
        if fall_prob > no_fall_prob:
            label = 'Fall'
            color = (0, 0, 255)  # Red for fall
            confidence = fall_prob
        else:
            label = 'No Fall'
            color = (0, 255, 0)  # Green for no fall
            confidence = no_fall_prob

        # Display results on the video frame
        cv2.putText(frame, f"{label}: {confidence:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show the frame with prediction
        cv2.imshow('Fall Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    model_path = "../models/fall_detection_model.h5"  # Path to the trained model
    real_time_fall_detection(model_path, video_source=0)  # Use webcam, or replace with video file path
