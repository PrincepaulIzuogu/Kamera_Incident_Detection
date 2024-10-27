import cv2
from openpose_setup import setup_openpose  # Import from the new module

def detect_falls(video_source):
    cap = cv2.VideoCapture(video_source)
    opWrapper = setup_openpose()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])
        
        cv2.imshow("OpenPose", datum.cvOutputData)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_source = 0
    detect_falls(video_source)
