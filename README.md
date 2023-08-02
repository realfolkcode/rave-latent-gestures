# RAVE Latent Space Exploration with Gestures

This is a pure Python project that allows users to navigate through the latent space of a pretrained [RAVE](https://github.com/acids-ircam/RAVE) model with gestures in real-time.

[Video example](https://youtu.be/nLMbJtdmUw4)

![image](https://github.com/realfolkcode/rave-latent-gestures/assets/64730991/72f6bcb5-a676-443f-b248-d57febe0a81e)

## How it works
The gesture encoder is designed in such a way that its latent codes follow the prior of RAVE (4-dimensional Gaussian distribution). Each time, RAVE decodes the gesture embeddings. More information is provided in the [training notebook](https://github.com/realfolkcode/rave-latent-gestures/blob/main/notebooks/rave_gesture_encoder.ipynb).

## Setup
1) Clone the repository
2) Install the required packages via `pip install -r requirements.txt` (tested with Python 3.10.12)
3) Download a [pretrained gesture encoder](https://github.com/realfolkcode/rave-latent-gestures/releases/download/v1.0.0/models-data.zip) and unzip it in the root directory of the project
4) Download the [MediaPipe HandLandmarker model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) and place it in the `models` directory
5) Move a pretrained RAVE model to the `models` directory (you can download some [here](https://acids-ircam.github.io/rave_models_download) or train your own custom model)

## Usage
1) Connect a webcam
2) Run `python generate.py --rave_model [PATH TO RAVE MODEL]`

Optional arguments:
- `--gesture_encoder` (the path to gesture encoder; change this to indicate your custom path or if you have trained a custom encoder)
- `--num_channels` (the number of output audio channels; depends on a RAVE model; default=1)
- `--num_blocks` (the number of streaming blocks; the smaller number corresponds to a smaller delay; default=4)
- `--temperature` (variance multiplier for encoder; indicates the randomness of sampling; default=2.0; recommendations: works fine from 1 to 4)
- `--cam_device` (the index of camera device; default=0)

## Acknowledgements
- Antoine Caillon and IRCAM for [RAVE](https://github.com/acids-ircam/RAVE)
- Google for [MediaPipe solutions](https://developers.google.com/mediapipe)
- Matthias Geier and other contributors for [sounddevice](https://github.com/spatialaudio/python-sounddevice/)
- Kapitanov et al. for [HaGRID dataset](https://github.com/hukenovs/hagrid)
