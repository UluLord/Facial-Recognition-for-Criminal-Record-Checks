import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


def load_model(facenet_model_path, weights_dir):
    """
    This function loads the pre-trained FaceNet Model and weights if specified.

    Args:
    - facenet_model_path (str): path to the pre-trained FaceNet Model file.
    - weights_dir (str): path to the pre-trained weights directory.
    
    Returns:
    - facenet_model (Keras model): The loaded FaceNet Model.
    """
    # Load the saved FaceNet model from JSON file
    if facenet_model_path.endswith(".json"):
        
        with open(facenet_model_path, "r") as json_file:
            loaded_model_json = json_file.read()
            
        facenet_model = tf.keras.models.model_from_json(loaded_model_json)
        
    # Load the saved FaceNet model from other file types
    else:
        facenet_model = tf.keras.models.load_model(facenet_model_path)
    
    # Load weights if specified
    if weights_dir is not None:
        facenet_model.load_weights(weights_dir)
    
    return facenet_model

def preprocess_image(face_image, img_size):
    """
    This function resizes, adds dimension, and normalizes a facial image using a pre-trained FaceNet model.
    
    Args:
    - face_image (array): the input facial image.
    - img_size (tuple): the desired size of the image after resizing.
    
    Returns:
    - face_image (array): the encoded facial image.
    """
    # Resize the facial image to the desired size.
    face_image = cv2.resize(face_image, img_size) 
    # Add a batch dimension to the image.
    face_image = np.expand_dims(face_image, axis=0)  
    # Convert the data type to float32.
    face_image = face_image.astype("float32")   
    # Normalize the pixel values to the range [0, 1].     
    face_image /= 255.0                              

    return face_image


def encoding_database(database_path, facenet_model, face_haarcascade_path, 
                      scale_factor, min_neighbors, min_size):
    """
    This function encodes a database of facial images using a pre-trained FaceNet model.
    
    Args:
    - database_path (str): path to the database file.
    - facenet_model (Tensorflow Model object): The pre-trained FaceNet model.
    - face_haarcascade_path (str): path to the Haar Cascade classifier XML file for face detection.
    - scale_factor (float): scale factor for the image pyramid in the face detection algorithm.
    - min_neighbors (int): minimum number of neighbors required for a detected face to be considered valid.
    - min_size (tuple): minimum size of a valid face.
        
    Returns:
    - database (dataframe): encoded database with a new column for the encoded facial images.
    """
    # Read the database from a CSV file.
    if database_path.endswith(".csv"):
        database = pd.read_csv(database_path)
        
    # Read the database from an Excel file.
    elif database_path.endswith(".xlsx"):
        database = pd.read_excel(database_path)
        
    # Raise TypeError if unsupported format specified
    else:
        raise TypeError("Unsupported database format: Only '.csv' or '.xlsx' format supported")
    
    # Initialize an empty list for the encoded images.
    encoded_images = []                              
    
    for image_path, id_num in zip(database["IMAGE PATH"], database["ID NUMBER"]):
        
        # Read the facial image from the file system.
        image = cv2.imread(image_path)
        # Load the Haar Cascade classifier for face detection.
        face_cascade = cv2.CascadeClassifier(face_haarcascade_path)
        # Detect faces in the image.
        faces = face_cascade.detectMultiScale(image, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)
        
        # Ensure that at least one face is detected in the image.
        assert len(faces)>0, f"Face cannot find in the image that belongs to {id_num} ID. Either adjust haarcascade parameters or upload different image"
        for (x, y, w, h) in faces:     
            # Extract the region of interest (ROI) containing the face.
            face_roi = image[y:y+h, x:x+w]    
            # Preprocess the facial image using the FaceNet model.
            face_roi_preprocessed = preprocess_image(face_roi, (160,160))
            # Encode the image and store the results.
            embedding = facenet_model.predict_on_batch(face_roi_preprocessed)
            sample_encoded_image = embedding / np.linalg.norm(embedding, ord=2)
            encoded_images.append(sample_encoded_image)
    
    # Add the encoded images to the database dataframe
    database["ENCODED IMAGE"] = encoded_images
    
    return database
