# Importing OpenCV Library for basic image processing functions
import os
import random
import win32api
import pygame
from imutils import face_utils
import dlib
import numpy as np
import cv2


# Numpy for array related functions
# Dlib for deep learning based Modules and face landmark detection
# face_utils for basic operations of conversion


pygame.mixer.init()
# alarm_sound = pygame.mixer.Sound('meme1_audio.wav')


# function to generate the video meme
#-------------------------------------------------------------------------------------#
def play_alert_video():
    # Get a list of all meme videos in the dataset folder
    dataset_folder = 'C:/Users/Prasanna/Documents/EDI6/memealert/memealert/meme'
    meme_videos = [file for file in os.listdir(
        dataset_folder) if file.endswith('.mp4')]

    # Choose a random meme video from the list
    meme_video_path = os.path.join(dataset_folder, random.choice(meme_videos))

    # Extract the audio file name by replacing the video extension with audio extension
    audio_file_name = os.path.splitext(meme_video_path)[0] + '.wav'

    # Check if the audio file exists
    if not os.path.exists(audio_file_name):
        print("Audio file not found for the meme video:", meme_video_path)
        return

    # Initialize Pygame mixer for audio playback
    pygame.mixer.init()

    # Load the audio file
    alarm_sound = pygame.mixer.Sound(audio_file_name)

    # Open the meme video file for reading
    cap = cv2.VideoCapture(meme_video_path)

    # Get the frame rate of the meme video
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Get the duration of the audio file
    audio_duration = alarm_sound.get_length()

    while cap.isOpened():
        ret, frame = cap.read()

        # Check if the video frame is successfully read
        if not ret:
            break

        # Display the meme video frame
        cv2.imshow('Meme', frame)

        # Play the audio
        pygame.mixer.Sound.play(alarm_sound)

        # Calculate the desired delay between frames
        delay = int(1000 / frame_rate)  # Delay in milliseconds

        # Synchronize the audio and video by adjusting the delay
        pygame.time.delay(int(delay * 0.8))

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture, stop the audio, and close the meme window
    cap.release()
    alarm_sound.stop()
    cv2.destroyWindow('Meme')
#-------------------------------------------------------------------------------------#


# Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)

# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# status marking for current state
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)


def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist


def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up/(2.0*down)

    # Checking if it is blinked
    if (ratio > 0.25):
        return 2
    elif (ratio > 0.21 and ratio <= 0.25):
        return 1
    else:
        return 0


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    # detected face in faces array
    face_frame = frame.copy()

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # The numbers are actually the landmarks which will show eye
        left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        # Now judge what to do for the eye blinks
        if (left_blink == 0 or right_blink == 0):
            sleep += 1
            drowsy = 0
            active = 0
            if (sleep > 6):
                status = "SLEEPING !!!"
                # alarm_sound.play()         #this was to generate the alarm sound
                # let we play the meme now
                play_alert_video()
                color = (255, 0, 0)

        elif (left_blink == 1 or right_blink == 1):
            sleep = 0
            active = 0
            drowsy += 1
            if (drowsy > 6):
                status = "Drowsy !"
                # for i in range(5):
                #     win32api.Beep(random.randint(37,10000), random.randint(750,3000))

                color = (0, 0, 255)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if (active > 6):
                status = "Active :)"
                color = (0, 255, 0)

        cv2.putText(frame, status, (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Result of detector", face_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
