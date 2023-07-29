import torch
import numpy as np
import sounddevice as sd
import cv2
import time
import mediapipe as mp
import argparse

from src.rave_wrapper import RaveWrapper
from src.landmark_detection import LandmarksWrapper

torch.set_grad_enabled(False)


def cv_loop(cap, lw):
    pTime = 0
    cTime = 0
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        h, w, c = img.shape

        img.flags.writeable = False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        img.flags.writeable = True

        lw.detector.detect_async(mp_image, int(time.time_ns() // 1000000))
        img = lw.draw_landmarks(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 3, (255, 255, 0), 3)

        cv2.imshow("Image", img)

        if cv2.waitKey(5) & 0xFF == 27:
            break


def main(args):
    rave_path = args.rave_model
    gesture_encoder_path = args.gesture_encoder
    channels = args.num_channels
    num_blocks = args.num_blocks
    temperature = args.temperature

    rave_model = torch.jit.load(rave_path).eval()

    cap = cv2.VideoCapture(0)
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape

    lw = LandmarksWrapper(h, w)

    gesture_encoder = torch.jit.load(gesture_encoder_path).eval()

    rw = RaveWrapper(rave_model, lw, gesture_encoder, channels=channels, num_blocks=num_blocks, temperature=temperature)

    with sd.OutputStream(blocksize=2048 * num_blocks,
                         channels=channels,
                         callback=rw.callback,
                         samplerate=48000,
                         dtype=np.float32):
        cv_loop(cap, lw)

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rave_model', type=str, required=True, help='path to RAVE model')
    parser.add_argument('--gesture_encoder', type=str, required=True, help='path to gesture encoder')
    parser.add_argument('--num_channels', type=int, required=False, default=1, help='number of output audio channels of RAVE model')
    parser.add_argument('--num_blocks', type=int, required=False, default=4, help='number of blocks')
    parser.add_argument('--temperature', type=float, required=False, default=2., help='variance multiplier for encoder')
    args = parser.parse_args()
    main(args)
