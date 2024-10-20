import cv2  # Import OpenCV for video and image processing
import numpy as np  # Import NumPy for numerical operations
import os  # Import OS for file handling
import csv  # Import CSV for handling CSV files
import time  # Import time for time-related functions
import pickle  # Import pickle for loading saved data
from sklearn.neighbors import KNeighborsClassifier  # Import KNN for classification
from datetime import datetime  # Import datetime for handling date and time

# Initialize video capture and face detector
video = cv2.VideoCapture(0)  # Capture video from the first camera device
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Load Haar Cascade for face detection

# Load the face labels and data from pickle files
with open('data/names.pkl', 'rb') as w:  # Open the pickle file containing names
    LABELS = pickle.load(w)  # Load names into the LABELS variable

with open('data/face_data.pkl', 'rb') as f:  # Open the pickle file containing face data
    FACES = pickle.load(f)  # Load face data into the FACES variable

# Debugging: Check the type and content of LABELS
print(f"Initial type of LABELS: {type(LABELS)}")  # Print the type of LABELS
print(f"Initial contents of LABELS: {LABELS}")  # Print the contents of LABELS

# Ensure LABELS is a list of strings
if isinstance(LABELS, dict):  # Check if LABELS is a dictionary
    LABELS = list(LABELS.values())  # Convert dictionary values to a list
elif not isinstance(LABELS, list):  # Check if LABELS is not a list
    raise ValueError("LABELS must be a list of strings.")  # Raise an error

# Convert all elements to strings if necessary
LABELS = [str(label) for label in LABELS]  # Ensure all labels are strings

# Debugging: Check types and contents after conversion
print(f"Type of LABELS after conversion: {type(LABELS)}")  # Print type after conversion
print(f"Contents of LABELS after conversion: {LABELS}")  # Print contents after conversion
print(f"Shape of FACES: {FACES.shape if isinstance(FACES, np.ndarray) else 'Not a numpy array'}")  # Print shape of FACES if it's a numpy array

# Make sure LABELS matches the number of faces
if FACES.shape[0] != len(LABELS):  # Check if the number of faces matches the number of labels
    raise ValueError(f"Number of faces ({FACES.shape[0]}) does not match number of labels ({len(LABELS)}).")  # Raise an error

# Initialize K-Nearest Neighbors (KNN) classifier
knn = KNeighborsClassifier(n_neighbors=5)  # Create KNN classifier with 5 neighbors
knn.fit(FACES, LABELS)  # Train the KNN model with face data and corresponding labels

# Load and resize the background image to match the display size (556x528)
imgbackground = cv2.imread("image.png")  # Read the background image
imgbackground = cv2.resize(imgbackground, (556, 528))  # Resize the image to fit the display

# Column names for the CSV attendance file
COL_NAMES = ['NAME_ID', 'TIME']  # Define column names for the attendance CSV

# Define the position and size of the area where the video frame will be displayed
video_frame_x = 40  # X-coordinate for the video frame position
video_frame_y = 120  # Y-coordinate for the video frame position
video_frame_width = 482  # Width of the video frame
video_frame_height = 365  # Height of the video frame

while True:
    ret, frame = video.read()  # Read a frame from the video capture
    if not ret:  # Check if the frame was captured successfully
        print("Failed to capture frame. Exiting...")  # Print an error message
        break  # Exit the loop

    # Resize the captured frame to fit the defined area
    frame = cv2.resize(frame, (video_frame_width, video_frame_height))  # Resize frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    faces = facedetect.detectMultiScale(gray, 1.3, 5)  # Detect faces in the grayscale image

    for (x, y, w, h) in faces:  # Loop through detected faces
        crop_img = frame[y:y+h, x:x+w]  # Crop the face from the frame
        resize_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)  # Resize and flatten the cropped image for prediction

        # Predict the person's name and ID using KNN
        output = knn.predict(resize_img)  # Get the prediction from the KNN model

        ts = time.time()  # Get the current timestamp
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%y")  # Format the date
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")  # Format the time

        attendance_file = f"Attendance/Attendance_{date}.csv"  # Define the attendance CSV file name
        exist = os.path.isfile(attendance_file)  # Check if the attendance file already exists

        # Draw a rectangle around the face and display the predicted name and ID
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y - 60), (x + w, y), (50, 50, 255), -1)  # Draw a background for text display
        
        # Handle the output for name and student ID
        if '_' in output[0]:  # Check if output contains an underscore
            name, student_id = output[0].split('_')  # Split name and student ID
        else:
            name = output[0]  # Use the output directly as name
            student_id = "Unknown"  # Default to "Unknown" if no ID is found

        # Display name and student ID on different lines
        cv2.putText(frame, f"Name: {name}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # Display name
        cv2.putText(frame, f"ID: {student_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # Display student ID

        # Prepare the attendance record
        attendance = [output[0], timestamp]  # Create attendance record with name and timestamp

    # Insert the resized video frame into the specified area of the background image
    imgbackground[video_frame_y:video_frame_y + video_frame_height, video_frame_x:video_frame_x + video_frame_width] = frame  # Place the video frame on the background
    cv2.imshow("frame", imgbackground)  # Show the combined image with video frame
    
    k = cv2.waitKey(1)  # Wait for a key press

    if k == ord('1'):  # Check if the '1' key is pressed
        time.sleep(2)  # Wait for 2 seconds

        # Write attendance to CSV file
        if exist:  # Check if the attendance file exists
            with open(attendance_file, "a", newline='') as csvfile:  # Open file in append mode
                writer = csv.writer(csvfile)  # Create a CSV writer object
                writer.writerow(attendance)  # Write attendance record
        else:  # If file does not exist
            with open(attendance_file, "a", newline='') as csvfile:  # Open file in append mode
                writer = csv.writer(csvfile)  # Create a CSV writer object
                writer.writerow(COL_NAMES)  # Write header names
                writer.writerow(attendance)  # Write attendance record

    if k == ord('q'):  # Check if the 'q' key is pressed
        break  # Exit the loop

# Release video capture and close all OpenCV windows
video.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows
