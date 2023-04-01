# **Facial Recognition for Criminal Record Checks**


Conventional criminal record checks by police involve manually searching through databases of criminal records to verify if an individual has a criminal record or not. The process can be time-consuming, inefficient, and prone to human error. Moreover, the information on these databases can become outdated quickly, which means the information accessed may not be up-to-date.

On the other hand, FaceNet, a face recognition technology, has the potential to check an individual's criminal record by analyzing their facial features. The technology is based on deep learning algorithms that can match a person's face with their criminal record, making the process much faster and more accurate.

One of the advantages of using FaceNet for criminal record checks is that it can be done remotely, which saves time and resources. This technology can also be used in real-time situations, such as identifying a suspect in a crime scene. Additionally, FaceNet has the ability to recognize faces even if the person has changed their appearance, such as growing a beard or wearing a disguise.

This project utilizes facial recognition technology to identify individuals with a criminal record using the FaceNet Model.

## **1. Usage**

To use the facial recognition system, follow these steps:

1. Clone the repo
2. Install the required packages
3. Prepare a Dataset
4. Download pretrained FaceNet Model
5. Run the main.py script

### *1.1 Clone the Repository:*

Clone the repository to your local machine. To clone the repository;

    git clone https://github.com/UluLord/Facial-Recognition-for-Criminal-Record-Checks.git

After cloning, change the directory, you are working, to this repository directory;

    cd Facial-Recognition-for-Criminal-Record-Checks

### *1.2 Requirements*

This project has been tested on these libraries;

* Tensorflow: 2.11.0
* OpenCV: 4.7.0
* Numpy: 1.22.4
* Pandas: 1.4.4
* Matplotlib: 3.7.0

To install the required packages, run the following command;

    pip install -r requirements.txt

**NOTE:** It may work with other versions of the libraries, but this has not been tested.

* This work has also been tested on NVIDIA GeForce RTX 3060 GPU.

**NOTE:** It is strongly recommended to work with a GPU.

### *1.3 Prepare a Dataset*

Prepare a database containing personnel informations like ID number, name, surname, image path (see database/database.xlsx for an example)

### *1.4 Specify Pretrained FaceNet Model*

Use **'facenet_model.json'** file, or download a pre-trained FaceNet model (e.g., from https://github.com/davidsandberg/facenet)

### *1.5 Run the *main.py* Script*

Run the **main.py** script with the appropriate arguments (see below for details)

***Parameters***

The **main.py** script accepts the following command line arguments:

  * **database_path:** Path to the database file containing face embeddings and labels. Default is './database/database.xlsx'.
  * **facenet_model_path:** Path to the pre-trained FaceNet Model. Default is 'facenet_model.json'.
  * **weights_dir:** Path to the pre-trained weights. Default is None.
  * **face_haarcascade_path:** Path to haarcascade frontal face detection classifier. Default is 'haarcascade_frontalface_default.xml'.
  * **source:** Video source for face recognition (0 for webcam, or path to video file). Default is 0.
  * **scale_factor:** Scale factor for face detection classifier. Default is 1.1.
  * **min_neighbors:** Minimum number of neighbors for face detection classifier. Default is 20.
  * **min_size:** Minimum size for face detection classifier (in pixels). Default is (30,30)
  * **dist_threshold:** Threshold distance for face recognition. Default is 0.12.
  * **save_to_path:** Directory path where the output video will be saved. Default is './recorded_videos'
  * **fps_rate:** Frame rate at which the output video will be saved. Default is 20.

***Example Usage***

To run the script with a database file located at **./database/database.xlsx**, a FaceNet Model from **facenet_model.json** with pre-trained **weights.h5**, and a video source from **a webcam**, use the following command:

    python main.py --database_path ./database/database.xlsx --facenet_model_path facenet_model.json --weights_dir weights.h5 --face_haarcascade_path haarcascade_frontalface_default.xml --source 0

## **2. Output**

The system outputs a video stream with faces highlighted and labeled if a match is found in the database. The video stream can be saved to a file by specifying the **save_to_path** with the FPS rate in **fps_rate** arguments.


***Sample Outputs***

* This footage shows a person who is not in the database.
    
    <img src="https://user-images.githubusercontent.com/99184963/228972072-ac92dc39-df88-4300-87a9-dec47064110d.png" width="600" height="400">

* This footage shows a person who is recorded in the dataset as 'WANTED'.
    
    <img src="https://user-images.githubusercontent.com/99184963/228972419-1e4134a8-47f1-4eb3-b4fe-50c6818bed3d.png" width="600" height="400">

* This footage shows the algorithm did not detect any face.
    
    <img src="https://user-images.githubusercontent.com/99184963/228972143-0493b78f-d5a2-4d3a-bb8d-19391b2b0061.png" width="600" height="400">

* This footage shows a person who is recorded in the dataset as 'PREVIOUSLY RECORDED'.
    
    <img src="https://user-images.githubusercontent.com/99184963/228972219-4161703e-7b16-4e8f-869f-b9722cf35410.png" width="600" height="400">
    

**NOTE-1:** The decimal number displayed above the images on the information screen does not represent the accuracy score but rather the distance between the detected face image and the image in the database.

**NOTE-2:** All the videos are available in the **'recorded_videos'** directory.

## **3. Citation**

If you use this repository in your work, please consider citing us as the following.

    @misc{ululord2023facial-recognition-for-criminal-record-checks,
	      author = {Fatih Demir},
          title = {Facial Recognition for Criminal Record Checks},
          date = {2023-04-01},
          url = {https://github.com/UluLord/Facial-Recognition-for-Criminal-Record-Checks}
          }


