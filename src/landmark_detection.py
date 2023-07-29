import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
import torch

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
 
MARGIN = 10 # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


class LandmarksWrapper:
    def __init__(self, height, width):
        options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='models/hand_landmarker.task'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=self.print_result)

        self.detector = HandLandmarker.create_from_options(options)
        self.hand_landmarks_list = []
        self.handedness_list = []

        self.height = height
        self.width = width

    # Create a hand landmarker instance with the live stream mode:
    def print_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.hand_landmarks_list = result.hand_landmarks
        self.handedness_list = result.handedness

    def draw_landmarks(self, output_image):
        # Loop through the detected hands to visualize.
        if not self.hand_landmarks_list:
            return output_image

        for idx in range(len(self.hand_landmarks_list)):
            hand_landmarks = self.hand_landmarks_list[idx]
            handedness = self.handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
            output_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = output_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(output_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        return output_image

    def get_center(self):
        if not self.hand_landmarks_list:
            return np.zeros(2)
        hand_landmarks = self.hand_landmarks_list[0]
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        center = np.stack((x_coordinates, y_coordinates)).mean(axis=-1)
        center[0] -= 0.5
        center[1] -= 0.5
        return center

    def _landmarks_to_torch(self, hand_landmarks):
        x = torch.tensor([landmark.x for landmark in hand_landmarks])
        y = torch.tensor([landmark.y for landmark in hand_landmarks]) * self.height / self.width
        z = torch.tensor([landmark.z for landmark in hand_landmarks])
        pos = torch.stack((x, y, z)).reshape(-1, 3)
        return pos
    
    def get_distance_vector(self):
        if not self.hand_landmarks_list:
            return None
        pos = self._landmarks_to_torch(self.hand_landmarks_list[0])
        pos = pos[:, :2]
        k = pos.shape[0]

        dist = torch.cdist(pos, pos, 2)
        dist = dist.masked_select(~torch.eye(k, dtype=bool)).view(1, -1)
        dist /= dist.max()
        return dist
    