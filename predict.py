import numpy as np
import tensorflow as tf
from preprocessing import preprocess_image


def predict_who_is(face_roi, database, facenet_model, dist_threshold):
    """
    This function predicts the identity of a person in the given face region of interest using a trained FaceNet model and a database of known face embeddings.
    
    Args:
    - face_roi (array): face region of interest.
    - database (dataframe): database of known face embeddings and labels.
    - facenet_model (Tensorflow Model object): FaceNet model.
    - dist_threshold (float): threshold distance for face recognition.
    
    Returns:
    - result_info (dict): dictionary containing the predicted identity and distance to the closest known face embedding in the database.
    """
    # Preprocess the face ROI image to the required shape for the model
    face_roi_preprocessed = preprocess_image(face_roi, (160,160))
    
    # Get the face embedding using the trained FaceNet model
    embedding = facenet_model.predict_on_batch(face_roi_preprocessed)
    
    # Normalize the embedding
    sample_encoded_image = embedding / np.linalg.norm(embedding, ord=2)
    
    # Initialize the distance to a large value
    distance = 100
    
    # Iterate over the database of known face embeddings
    for id_num, db_encoded_image in zip(database["ID NUMBER"], database["ENCODED IMAGE"]):
        
        # Calculate the distance between the face embedding and each known face embedding in the database
        encode_image_distance = np.linalg.norm(tf.square(tf.subtract(sample_encoded_image, db_encoded_image)))
        
        # If the distance is smaller than the current minimum distance, update the minimum distance and predicted identity
        if encode_image_distance < distance:
            distance = encode_image_distance
            id_number = id_num
    
    # If the minimum distance is greater than the distance threshold, the identity is unknown
    if distance > dist_threshold:
        id_number = None
    
    # Store the predicted identity and distance in a dictionary
    result_info = {"id_number": id_number,
                   "distance": distance}
        
    return result_info
