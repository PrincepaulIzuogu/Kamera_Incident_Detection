import cv2
from openpose_setup import setup_openpose  # Import from the new module

def detect_falls(video_source=0):  # Default to 0 for the first webcam
    cap = cv2.VideoCapture(video_source)
    opWrapper = setup_openpose()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])
        
        # Display the results
        cv2.imshow("OpenPose", datum.cvOutputData)

        # Placeholder for fall detection logic based on keypoints
        # Add your fall detection logic here

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_falls(0)  # Use 0 for the default webcam
