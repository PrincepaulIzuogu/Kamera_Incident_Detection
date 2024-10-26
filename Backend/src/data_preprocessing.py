import cv2
import os

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Save frame as image file
        cv2.imwrite(os.path.join(output_folder, f'frame_{count:04d}.jpg'), frame)
        count += 1
    
    cap.release()
    print(f"Extracted {count} frames from {video_path}")

if __name__ == "__main__":
    # Directory containing videos
    video_directory = "/app/data/fall_detection/videos/"  # Update this path if necessary
    output_directory = "/app/frames/fall/"  # Ensure this path is correct

    # Iterate over all video files in the directory
    for filename in os.listdir(video_directory):
        if filename.endswith(('.mp4', '.avi', '.mov')):  # Add any other video formats you want to support
            video_path = os.path.join(video_directory, filename)
            extract_frames(video_path, output_directory)
