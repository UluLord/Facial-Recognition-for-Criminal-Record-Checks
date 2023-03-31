import cv2
import numpy as np
import pandas as pd

class ScreenProcess:
    """
    This class processes an image to display information about a person.
    """
    def __init__(self, database):
        """
        Initialize the class.
        
        Args:
        - database (dataframe): contains the personnel database.
        """
        self.database = database
        
        # Set the window size and its position on the screen
        self.screen_height = 900
        self.screen_width = 1200
        self.split_line = 850


    def show_status(self, screen, database, id_number, note):
        """
        This function updates the status of an individual based on their ID number and return the updated status to be displayed on the screen.

        Args:
        - screen (array): main screen.
        - database (dataframe): contains the personnel database.
        - id_number (int or None): the ID number of the personnel.
        - note (str or None): a note on the status of face detection.

        Returns:
        - screen (array): updated status to be displayed on the screen.
        """
        # Define the coordinates of the rectangle on the screen
        x1 = self.split_line
        y1 = self.screen_height - 200
        x2 = self.screen_width
        y2 = self.screen_height
            
        # Get the columns "STATUS" and "CHARGE" from the database
        status_and_charge = database[["STATUS", "CHARGE"]].columns
    
        # Iterate through the columns "STATUS" and "CHARGE"
        for col in status_and_charge:
        
            # Check if the ID number is not None
            if id_number is not None:
                # Get the information for the given ID number
                get_info = database[col][database["ID NUMBER"] == id_number].values
            
                # Update the status on the screen if the column is "STATUS"
                if col == "STATUS":
                    if get_info[0].upper() == "WANTED":
                        cv2.rectangle(screen, (x1, y1), (x2, y2), (0,0,255), -1) 
                        cv2.putText(screen, get_info[0].upper(), (x1 + 90, y1 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                    elif get_info[0].upper() == "PREVIOUSLY CONVICTED":
                        cv2.rectangle(screen, (x1, y1), (x2, y2), (50, 200, 255), -1) 
                        cv2.putText(screen, get_info[0].upper(), (x1 + 10, y1 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
                # Update the charge on the screen if the column is "CHARGE"
                elif col == "CHARGE":
                    cv2.putText(screen, col.title()+":", (x1 + 10, y2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(screen, str(get_info[0]), (x1 + 125, y2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
            # If the ID number is None, 
            else:
                # Pass the status if the note argument is 'No face detected' 
                if note == "No face detected":
                    pass
                # Update the status to "NO CRIMINAL RECORD" if the note argument is not 'No face detected'
                else:
                    cv2.rectangle(screen, (x1, y1), (x2, y2), (255, 0, 0), -1) 
                    cv2.putText(screen, "NO CRIMINAL RECORD", (x1 + 20, y1 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)        
        
        return screen


    def personal_info(self, screen, id_number, note):
        """
        This function displays personal informations on the screen including ID number, name, date of birth, etc. 

        Args:
        - screen (array): main screen.
        - id_number (str or None): the ID number of the personnel.
        - note (str or None): a note on the status of face detection.

        Returns:
        - self.show_status(screen, database_copy, id_number, w, note): displays the status of the personnel's information.
        """
        # Initialize dimensions for text placement
        w = 20
        h = 50
        y = 220
    
        # Create a copy of the database
        database_copy = self.database.copy()
    
        # Extract the year, month, and day information into separate columns
        database_copy["BIRTH DATE"] = pd.to_datetime(database_copy["BIRTH DATE"]).dt.strftime("%Y/%m/%d")
        
        # Drop the irrelevant columns
        db_dropped = database_copy.drop(["IMAGE PATH", "ENCODED IMAGE", "STATUS", "CHARGE"], axis=1)
        columns = db_dropped.columns
    
        # Loop through the columns of the database
        for col in columns:
            # Display column title on screen
            cv2.putText(screen, col.title(), (self.split_line + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
            # If an ID number is provided, display the corresponding information
            if id_number is not None:
                get_info = db_dropped[col][db_dropped["ID NUMBER"] == id_number].values
                cv2.putText(screen, ": " + str(get_info[0]), (self.split_line + 180, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            else:
                # If no ID number is provided, display "N/A" or blank depending on note
                if note == "No face detected":
                    cv2.putText(screen, ":", (self.split_line + 180, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                else:
                    cv2.putText(screen, ": N/A", (self.split_line + 180, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
            # Add separator between each column
            cv2.putText(screen, "-"*40, (self.split_line + w, y + h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Increase the y-axis value for the next column
            h += 35

        return self.show_status(screen, database_copy, id_number, note)
    
    
    def info_screen(self, screen, personnel_image_path, id_number, distance, note):
        """
        This function displays information about personnel on the screen.
    
        Args:
        - screen (array): a black canvas on which the screen will be drawn.
        - personnel_image_path (str or None): path to personnel image.
        - id_number (str): ID number of the personnel.
        - distance (float): distance score.
        - note (str or None): a note on the status of face detection.
    
        Returns:
        - self.personal_info(screen, id_number, note): shows the personnel informations.
        """
        # Define the location of the information box on the screen.
        info_middle_point = (self.screen_width - self.split_line)//2
        info_middle = 850 + info_middle_point
        x1 = info_middle - (info_middle_point//2)
        x2 = info_middle + (info_middle_point//2)
        y1 = 90 
        y2 = 220
    
        # Calculate the size of the personnel image.
        frame_width = x2 - x1
        frame_height = y2 - y1
    
        # If the ID number is not None, display the personnel information.
        if id_number is not None:
            # Load and resize the personnel image.
            personnel_image = cv2.imread(personnel_image_path)
            personnel_image = cv2.resize(personnel_image, dsize=(frame_width, frame_height))
        
            # Add the personnel image to the screen.
            screen[y1:y2, x1:x2] = personnel_image
        
            # Add the distance to the screen.
            cv2.rectangle(screen, (x1, y1-20), (x1+40, y1-5), (255, 255, 255), -1)
            cv2.putText(screen, str(np.round(distance, 4)), (x1 + 2, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)

        # If the ID number is None, display a note.
        else:
            cv2.rectangle(screen, (x1, y1), (x2, y2), (255,255,255), 2)
        
            if note == "No face detected":
                cv2.putText(screen, note, (x1+15, y1+70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        
            elif note == "Not in the database":
                cv2.putText(screen, note, (x1+8, y1+70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        return self.personal_info(screen, id_number, note)


    def live_screen(self, screen, frame):
        """
        This function adds a live frame to the screen.

        Args:
        - screen (array): main screen.
        - frame (array): a frame from the live video feed.

        Returns:
        - screen (array): merged screen.
        """
        # Set the frame height and width to match the screen size
        frame_height = self.screen_height
        frame_width = self.split_line
        
        # Resize the frame to fit the screen
        frame = cv2.resize(frame, dsize=(frame_width, frame_height))
        
        # Copy the frame onto the left side of the screen
        screen[:frame_height, :frame_width] = frame
        
        return screen
    
    
    def __call__(self, frame, personnel_image_path, id_number, distance, note):
        """
        Calling this class merges the live video display with personnel information and returns the resulting screen.

        Args:
        - frame (array): representing an image frame.
        - personnel_image_path (str or None): path to personnel image.
        - id_number (str or None): the ID number of the personnel.
        - distance (float or None): distance score.
        - note (str or None): a note on the status of face detection.

        Returns:
        - screen (array): updated screen.
        """
        # Create a black screen.
        black_canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype="uint8")

        # Get personnel info and display on screen
        info_result = self.info_screen(black_canvas, personnel_image_path, id_number, distance, note)

        # Get live video display and merge with personnel info
        live_result = self.live_screen(info_result, frame)

        # Set the merged result as the output screen
        screen = live_result

        return screen
