import argparse
from preprocessing import load_model, encoding_database
from face_recognition import FaceRecognition

def set_arguments():
    """
    This function parses command line arguments and returns them as a dictionary. 
    """ 
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(
            # Description of the project
            description="This project implements facial recognition to identify guilty persons. \n\nAdjust the parameters if necessary:",
            # Usage string to display
            usage="Facial Recognition",
            # Set the formatter class to ArgumentDefaultsHelpFormatter
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            # Set prefix chars
            prefix_chars="-",
            # Set default value for argument_default
            argument_default=argparse.SUPPRESS,
            # Allow abbreviations of options
            allow_abbrev=True,
            # Add help argument
            add_help=True)
    
    # Add arguments
    parser.add_argument('--database_path', type=str, default='./database/database.xlsx', required=False,
                        help='Path to the database file containing face embeddings and labels')
    
    parser.add_argument('--facenet_model_path', type=str, default='facenet_model.json', required=False,
                        help='Path to the pre-trained FaceNet Model')
    
    parser.add_argument('--weights_dir', type=str, default=None, required=False,
                        help='Path to the pre-trained weights')
    
    parser.add_argument('--face_haarcascade_path', type=str, default='haarcascade_frontalface_default.xml', required=False,
                        help='Path to haarcascade frontal face detection classifier')

    parser.add_argument('--source', type=str, default=0, required=False,
                        help='Video source for face recognition (0 for webcam, or path to video file)')
    
    parser.add_argument('--scale_factor', type=float, default=1.1, required=False,
                        help='Scale factor for face detection classifier')
    
    parser.add_argument('--min_neighbors', type=int, default=20, required=False,
                        help='Minimum number of neighbors for face detection classifier')
    
    parser.add_argument('--min_size', type=tuple, default=(30, 30), required=False,
                        help='Minimum size for face detection classifier (in pixels)')
    
    parser.add_argument('--dist_threshold', type=float, default=0.12, required=False,
                        help='Threshold distance for face recognition')
    
    parser.add_argument('-save_to_path', type=str, default="./recorded_videos", required=False,
                        help='directory path where the output video will be saved.')
    
    parser.add_argument('-fps_rate', type=float, default=20, required=False,
                        help='Frame rate at which the output video will be saved,')
    
    # Parse the arguments and convert them to a dictionary
    args = vars(parser.parse_args())
    
    return args

    
if __name__ == "__main__":
    
    # Parse command line arguments
    args = set_arguments()
    
    # Load pre-trained FaceNet Model and weights if specified
    facenet_model = load_model(args["facenet_model_path"], args["weights_dir"])
        
    # Encode database
    database = encoding_database(args["database_path"], facenet_model, args["face_haarcascade_path"],
                                 args["scale_factor"], args["min_neighbors"], args["min_size"])

    # Initialize FaceRecognition object
    face_recognition = FaceRecognition(database, facenet_model, args["face_haarcascade_path"])
    
    # Start face detection and recognition
    face_recognition(args["source"], args["scale_factor"], args["min_neighbors"], args["min_size"], args["dist_threshold"],
                     args["save_to_path"], args["fps_rate"])
