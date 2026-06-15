import cv2
from pose_estimator import PoseEstimator

pose = PoseEstimator()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results, landmarks = pose.detect_pose(frame)
    frame = pose.draw_landmarks(frame, results)
    cv2.imshow('Pose Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
