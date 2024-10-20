import cv2  # Import OpenCV for video and image processing
import numpy as np  # Import NumPy for numerical operations
import os  # Import OS for file handling
import pickle  # Import pickle for saving data

# Initialize video capture from the webcam
video = cv2.VideoCapture(0)  # Capture video from the first camera device
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # Load Haar Cascade for face detection

face_data = []  # Initialize an empty list to store face images
i = 0  # Initialize a counter for the loop

# Get the name of the student
name = input("Enter your name: ")  # Prompt user for their name
# Get the student ID separately
student_id = input("Enter your student ID: ")  # Prompt user for their student ID
identifier = f"{name}_{student_id}"  # Combine name and ID as a unique identifier

while True:
    ret, frame = video.read()  # Capture a frame from the webcam
    if not ret:  # Check if the frame was captured successfully
        print("Failed to capture frame. Exiting...")  # Print error message
        break  # Exit the loop

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale for face detection
    faces = facedetect.detectMultiScale(gray, 1.3, 5)  # Detect faces in the grayscale image

    for (x, y, w, h) in faces:  # Loop through detected faces
        crop_img = frame[y:y+h, x:x+w]  # Crop the face from the frame
        resize_img = cv2.resize(crop_img, (50, 50))  # Resize the cropped face image to 50x50 pixels
        
        # Collect face images at specific intervals
        if len(face_data) < 100 and i % 10 == 0:  # Capture only up to 100 face images, every 10th frame
            face_data.append(resize_img)  # Add the resized image to the face_data list
            cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)  # Display the number of captured images
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)  # Draw a rectangle around the detected face

        cv2.imshow("Frame", frame)  # Show the frame with detected faces
        i += 1  # Increment the frame counter

    if len(face_data) >= 100:  # Check if we have captured enough face images
        break  # Exit the loop if we have 100 images

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if the 'q' key is pressed
        break  # Exit the loop

video.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows

# Convert face_data into a numpy array
face_data = np.array(face_data)  # Convert the list of face images to a NumPy array
face_data = face_data.reshape(100, -1)  # Reshape the array to have 100 samples with flattened image data

# Create 'data/' directory if it doesn't exist
if not os.path.exists('data'):  # Check if the 'data' directory exists
    os.makedirs('data')  # Create the 'data' directory

# Save the name-ID and face data using pickle
if 'names.pkl' not in os.listdir('data/'):  # Check if names.pkl exists in the data directory
    names = [identifier] * 100  # Create a list with the identifier repeated 100 times
    with open('data/names.pkl', 'wb') as f:  # Open names.pkl in write-binary mode
        pickle.dump(names, f)  # Save the names list to the file
else:  # If names.pkl already exists
    with open('data/names.pkl', 'rb') as f:  # Open names.pkl in read-binary mode
        names = pickle.load(f)  # Load existing names from the file
    names = names + [identifier] * 100  # Append the new identifier repeated 100 times
    with open('data/names.pkl', 'wb') as f:  # Open names.pkl in write-binary mode
        pickle.dump(names, f)  # Save the updated names list to the file

if 'face_data.pkl' not in os.listdir('data/'):  # Check if face_data.pkl exists in the data directory
    with open('data/face_data.pkl', 'wb') as f:  # Open face_data.pkl in write-binary mode
        pickle.dump(face_data, f)  # Save the face data to the file
else:  # If face_data.pkl already exists
    with open('data/face_data.pkl', 'rb') as f:  # Open face_data.pkl in read-binary mode
        faces = pickle.load(f)  # Load existing face data from the file
    faces = np.append(faces, face_data, axis=0)  # Append the new face data to the existing data
    with open('data/face_data.pkl', 'wb') as f:  # Open face_data.pkl in write-binary mode
        pickle.dump(faces, f)  # Save the updated face data to the file

print("Data saved successfully.")  # Print success message
