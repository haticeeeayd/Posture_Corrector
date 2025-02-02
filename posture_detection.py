import numpy as np
import cv2
import mediapipe as mp
import pygame  
import dlib
from scipy.spatial import distance as dist


def vector_angle_3d(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  
    angle = np.arccos(cos_theta)
    return np.degrees(angle)


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def play_sound(file_name):
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(file_name)
        pygame.mixer.music.play()


def stop_sound():
    if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2, static_image_mode=False,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  


face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20  
frame_counter = 0  
sleep_detected = False


cap = cv2.VideoCapture(0)
good_posture_baseline = None  

angle_threshold = 15
z_diff_threshold = 0.1
bad_posture_warning_triggered = False  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                landmarks[mp_pose.PoseLandmark.NOSE.value].y]
        neck = [(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x +
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2,
                (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y +
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2]

        nose_z = landmarks[mp_pose.PoseLandmark.NOSE.value].z
        left_shoulder_z = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
        right_shoulder_z = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z

        neck_to_nose_vector_3d = [
            nose[0] - neck[0],
            nose[1] - neck[1],
            nose_z - ((left_shoulder_z + right_shoulder_z) / 2)
        ]
        vertical_vector_3d = [0, -1, 0]
        shoulder_z_difference = abs(left_shoulder_z - right_shoulder_z)

        if good_posture_baseline is None:
            good_posture_baseline = {
                "shoulder_z_diff": shoulder_z_difference,
                "neck_angle_3d": vector_angle_3d(neck_to_nose_vector_3d, vertical_vector_3d)
            }
            print("Good posture baseline set:", good_posture_baseline)

        neck_flexion_angle_3d = vector_angle_3d(neck_to_nose_vector_3d, vertical_vector_3d)
        bad_posture = False
        if abs(neck_flexion_angle_3d - good_posture_baseline["neck_angle_3d"]) > angle_threshold:
            bad_posture = True
        if abs(shoulder_z_difference - good_posture_baseline["shoulder_z_diff"]) > z_diff_threshold:
            bad_posture = True

        height, width, _ = frame.shape
        cv2.putText(frame, f"Neck Angle (3D): {neck_flexion_angle_3d:.2f} degrees",
                    (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)

        if bad_posture:
            cv2.putText(frame, "BAD POSTURE", (int(width * 0.05), int(height * 0.9)),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            if not bad_posture_warning_triggered:
                play_sound('dikdur.mp3')  
                bad_posture_warning_triggered = True
        else:
            cv2.putText(frame, "GOOD POSTURE", (int(width * 0.05), int(height * 0.9)),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            bad_posture_warning_triggered = False  

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    
    faces = face_detector(gray_frame)
    for face in faces:
        landmarks = landmark_predictor(gray_frame, face)

        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= CONSEC_FRAMES:
                cv2.putText(frame, "SLEEPING WARNING!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                play_sound('bad_posture_warning.mp3')
        else:
            frame_counter = 0
            stop_sound()

    cv2.imshow("Posture and Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()