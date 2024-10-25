import os
import cv2

def extract_frames_from_all_videos(input_folder, output_folder, label):
    """Extracts frames from all video files in the input folder and stores them under the appropriate label."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi'))]

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        video_output_folder = os.path.join(output_folder, label, os.path.splitext(video_file)[0])

        if not os.path.exists(video_output_folder):
            os.makedirs(video_output_folder)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_file = os.path.join(video_output_folder, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(frame_file, frame)
            frame_count += 1

        cap.release()
        print(f"Extracted {frame_count} frames from {video_file}.")

if __name__ == "__main__":
    # Paths to the fall and no-fall videos
    fall_videos_dir = "../data/fall_detection"
    no_fall_videos_dir = "../data/no_fall_detection"

    # Output directories for frames
    output_dir = "../data/frames"

    # Extract frames for falls
    extract_frames_from_all_videos(fall_videos_dir, output_dir, label='fall')

    # Extract frames for no-falls
    extract_frames_from_all_videos(no_fall_videos_dir, output_dir, label='no_fall')
