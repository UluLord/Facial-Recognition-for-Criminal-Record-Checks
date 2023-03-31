import os
import cv2
import time
from predict import predict_who_is
from screen_process import ScreenProcess

class FaceRecognition:
    """
    This class provides functionalities to detect faces and recognize personnel information from an image.
    """
    def __init__(self, database, facenet_model, face_haarcascade_path):
        """
        Initialize the class.
        
        Args:
        - database (dataframe): contains the personnel database.
        - facenet_model (Tensorflow Model object): the pre-trained Facenet model for face recognition.
        - face_haarcascade_path (str): path to the haarcascade file for face detection.
        """
        self.database = database
        self.facenet_model = facenet_model
        self.face_haarcascade_path = face_haarcascade_path
        self.ScreenProcess = ScreenProcess(database)

        # Set the window size and its position on the screen
        self.screen_height = 900
        self.screen_width = 1200
        self.split_line = 850
        
    @staticmethod
    def compute_fps(fps_frame_count, fps_start_time):
        """
        This function computes and returns the frames per second.
        
        Args:
        - fps_frame_count (int): number of frames.
        - fps_start_time (float): start time.
        
        Returns:
        - fps (int): the frames per second
        """
        fps_frame_count += 1
        fps_end_time = time.time()
        fps = round(fps_frame_count / (fps_end_time - fps_start_time))
        
        return fps

    @staticmethod
    def specify_filename(path):
        """
        This function checks if a file with that name already exists in the path.
        If it does, adds a suffix to the filename to make it unique, then returns the new filename.
        If it does not, returns the original filename.

        Args:
        - path (str): path where the output file should be saved.

        Returns:
        - new_filename (str): new filename for the output file.
        """
        filename = "output_video.mp4"
        full_path = os.path.join(path, filename)

        # check if the file already exists
        if os.path.exists(full_path):
            # add a suffix to the filename to make it unique
            suffix = 1
            while True:
                # create a new filename with a suffix
                new_filename = f"{os.path.splitext(filename)[0]}_{suffix}.mp4"
                full_path = os.path.join(path, new_filename)

                # check if the new filename already exists
                if not os.path.exists(full_path):
                    break
                suffix += 1
        else:
            new_filename = filename

        return new_filename


    def __call__(self, source, scale_factor, min_neighbors, min_size, dist_threshold, 
                 save_to_path, fps_rate):
        """
        Calling this class runs the facial recognition program on a video stream from a source.

        Args:
        - source (str): video stream source. Can be an integer value or a string representing a file path.
        - scale_factor (float): scale factor for the image pyramid in the face detection algorithm.
        - min_neighbors (int): minimum number of neighbors required for a detected face to be considered valid.
        - min_size (tuple): minimum size of a valid face.
        - dist_threshold (float): distance threshold for face recognition.
        - save_to_path (str): path where the output video will be saved.
        - fps_rate (float): frame rate at which the output video will be saved,

        Returns:
        - None
        """
        # Convert 0 to integer
        if source == "0":
            source = 0
            
        # Open the video capture device
        cap = cv2.VideoCapture(source)
    
        # Load the face detection classifier
        face_cascade = cv2.CascadeClassifier(self.face_haarcascade_path)
        
        if source == 0:
            # Check if the camera was successfully opened
            if cap.isOpened():
                print("Camera is open")
            # Raise IOError if the camera was not opened 
            else:
                cap.release()
                cv2.destroyAllWindows()
                raise IOError("Could not open camera")
        
        # Define filename, the codec and create VideoWriter object to save the video
        if os.path.exists(save_to_path) == False:
            os.makedirs(save_to_path)
        filename = self.specify_filename(save_to_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"./{save_to_path}/{filename}", fourcc, fps_rate, (self.screen_width, self.screen_height))

        # Initialize variables for FPS calculation
        fps_frame_count = 0
        fps_start_time = time.time()
    
        # Loop through the video stream
        while True:
            # Read a frame from the video stream
            ret, frame = cap.read()

            if not ret:
                break

            # Flip the frame horizontally if the video source is 0
            if source==0:
                frame = cv2.flip(frame, 1)

            # Detect faces in the grayscale frame
            faces = face_cascade.detectMultiScale(frame, 
                                                  scaleFactor=scale_factor, 
                                                  minNeighbors=min_neighbors, 
                                                  minSize=min_size, 
                                                  flags=cv2.CASCADE_SCALE_IMAGE)
        
            # If no face is detected, display a message on the screen
            screen = self.ScreenProcess(frame, None, None, None, note="No face detected")
            
            # For each detected face
            for (x, y, w, h) in faces:
                # Get face roi
                face_roi = frame[y:y+h, x:x+w]
                # Draw a rectangle and the predicted expression label on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Use a trained face recognition model to identify the person in the face ROI
                face_info = predict_who_is(face_roi, self.database, self.facenet_model, dist_threshold)

                # If the person is in the database, display their information on the screen
                if face_info["id_number"] is not None:
                    id_number = face_info["id_number"]
                    distance = face_info["distance"]
                    personnel_image_path = self.database["IMAGE PATH"][self.database["ID NUMBER"] == id_number].values
                    screen = self.ScreenProcess(frame, personnel_image_path[0], id_number, distance, None)
            
                # If the person is not in the database, display a message on the screen
                else:
                    screen = self.ScreenProcess(frame, None, None, None, note="Not in the database")

            # Calculate and display FPS on the video window
            fps = self.compute_fps(fps_frame_count, fps_start_time)
            cv2.putText(screen, f"FPS: {fps}", (self.split_line + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            fps_start_time = time.time()
            fps_frame_count += 0
            
            # Write the frame to the output video file
            out.write(screen)

            # Show the frame
            cv2.imshow("Face Recognition", screen)

            # Exit if 'ESC' key is pressed
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
        print(f"Video saved to '{save_to_path}/{filename}'")
        # Release everything and destroy all windows if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()