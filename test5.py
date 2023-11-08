import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame
import random
import os
from moviepy.editor import VideoFileClip
######################################################################################
def play_alert_video():
    # Get a list of all meme videos in the dataset folder
    dataset_folder = 'C:/Users/Prasanna/Documents/EDI6/memealert/memealert/meme'
    meme_videos = [file for file in os.listdir(dataset_folder) if file.endswith('.mp4')]

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
    pygame.mixer.music.load(audio_file_name)

    # Play the audio in a loop
    pygame.mixer.music.play(-1)

    # Open the meme video file for reading
    cap = cv2.VideoCapture(meme_video_path)

    # Get the frame rate of the meme video
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()

        # Check if the video frame is successfully read
        if not ret:
            break

        # Display the meme video frame
        cv2.imshow('Meme', frame)

        # Calculate the desired delay between frames
        delay = int(1000 / frame_rate)  # Delay in milliseconds

        # Synchronize the audio and video by adjusting the delay
        pygame.time.delay(int(delay * 0.8))

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Stop the audio and release resources
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    cap.release()
    cv2.destroyWindow('Meme')

###########################################################################

def start_process():
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    # Initialize the face detector and landmark detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Status marking for current state
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
        ratio = up / (2.0 * down)

        # Checking if it is blinked
        if ratio > 0.25:
            return 2
        elif ratio > 0.21 and ratio <= 0.25:
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

            cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_blink = blinked(landmarks[36], landmarks[37],
                                 landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43],
                                  landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            if left_blink == 0 or right_blink == 0:
                sleep += 1
                drowsy = 0
                active = 0
                if sleep > 6:
                    status = "SLEEPING !!!"
                    play_alert_video()
                    color = (255, 0, 0)

            elif left_blink == 1 or right_blink == 1:
                sleep = 0
                active = 0
                drowsy += 1
                if drowsy > 6:
                    status = "Drowsy !"
                    color = (0, 0, 255)

            else:
                drowsy = 0
                sleep = 0
                active += 1
                if active > 6:
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

    cap.release()
    cv2.destroyAllWindows()
#############################################################################################
def enable_camera():
    messagebox.showinfo("Message", "Camera enabled")
    start_process()

# Create the main window
window = tk.Tk()
window.title("sleepliness detection")
window.geometry("400x200")

# Create the buttons
camera_button = tk.Button(window, text="Get Ready for focus mode ⚡", command=enable_camera)
camera_button.pack(pady=20)

# start_button = tk.Button(window, text="Start Process", command=start_process)
# start_button.pack(pady=10)

# Run the GUI main loop
window.mainloop()