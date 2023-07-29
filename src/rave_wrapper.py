import numpy as np
import torch

class RaveWrapper():
    def __init__(self, model, hand_tracker, gesture_encoder, channels, num_blocks, temperature):
        self.model = model
        self.hand_tracker = hand_tracker
        self.gesture_encoder = gesture_encoder
        self.channels = channels
        self.num_blocks = num_blocks
        self.temperature = temperature
        self.mu = torch.zeros(1, 4, self.num_blocks)
        self.log_var = torch.zeros(1, 4, self.num_blocks)

    @torch.no_grad()
    def callback(self, outdata, frames, t, status):
        dist = self.hand_tracker.get_distance_vector()
        if dist is not None:
            mu, log_var = self.gesture_encoder(dist)
            self.hand_tracker.mu = mu[:, :, None].repeat(1, 1, self.num_blocks)
            self.log_var = log_var[:, :, None].repeat(1, 1, self.num_blocks)
        eps = torch.randn(4, self.num_blocks)
        var = torch.exp(0.5 * self.log_var)
        latent = self.mu + self.temperature * var * eps
        data = self.model.decode(latent).reshape(-1, self.channels).numpy()[:frames]
        outdata[:] = data
