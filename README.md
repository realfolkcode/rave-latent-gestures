# RAVE Latent Space Exploration with Gestures

This is a pure Python project that allows users to navigate through the latent space of a pretrained RAVE model with gestures.

## Setup
1) Clone the repository
2) Install the required packages via `pip install -r requirements.txt` (tested with Python 3.10.12)
3) Download a [pretrained gesture encoder](https://github.com/realfolkcode/rave-latent-gestures/releases/download/v1.0.0/models-data.zip) and unzip it in the root directory of the project
4) Download the [MediaPipe HandLandmarker model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) and place it in the `models` directory
5) Move a pretrained RAVE model to the `models` directory (you can download some [here](https://acids-ircam.github.io/rave_models_download) or train your own custom model)
