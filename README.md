# HELMET VEST HEAD DETECTION USING YOLOv8

## Image-Based Prediction Model

This repository houses an image-based prediction model developed using Python, Flask, and HTML languages. The model is capable of detecting helmets, vests, and heads within images using a custom-trained variant of YOLOv8.

## Model Details

The code leverages YOLOv8, a pre-trained model, which was further fine-tuned on a custom dataset.

Two different trained YOLOv8 models are employed to predict images effectively.

## Deployment

The model is deployed using Flask.

During inference, a "run" folder is automatically generated to store predicted images.

To manage memory efficiently, the predicted images have been relocated to the HTML static folder, and unnecessary folders within the "run/detect" folder have been removed.

Feel free to customize this setup to suit your specific requirements.

## HTML Templates

This repository includes a "template" folder that stores HTML files:

"image.html" allows users to upload an image.

"predict.html" displays the predicted images.

## File Upload

The code currently supports uploading a single file, but it can be easily modified to handle multiple files.

## Dataset Source

The dataset used for training this model is sourced from Kaggle at the following URL:  https://www.kaggle.com/datasets/muhammetzahitaydn/hardhat-vest-dataset-v3

## Dependencies

Ensure you have the following dependencies installed before running the code:

	!pip install ultralytics==8.0.20
	
	!pip install flask
	
	!pip uninstall opencv-python
	
	!pip install opencv-python-headless

## How to run

To run the model, execute the following command in your terminal:

	python3 main.py
