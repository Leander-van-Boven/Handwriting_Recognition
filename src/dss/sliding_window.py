import cv2 as cv
import numpy as np
from tqdm import tqdm


class SlidingWindowClassifier:
    def __init__(self, model, num_classes, word_images, conf):
        self.model = model
        self.num_classes = num_classes
        self.word_images = word_images
        self.window_size = conf.window.size
        self.hop_size = conf.window.hop_size

    def resize_and_slice(self, image):
        assert self.hop_size > 0

        ratio = self.window_size[0] / image.shape[0]
        new_width = int(image.shape[1] * ratio)
        resized_im = cv.resize(image, (new_width, self.window_size[0]))

        image_width = resized_im.shape[1]
        window_width = self.window_size[1]
        num_steps, mod = divmod((image_width - window_width), self.hop_size)
        slices = [
            np.s_[:, pos:(pos + window_width), :]
            for pos in range(0, (num_steps * self.hop_size) + 1, self.hop_size)
        ]
        if mod != 0:
            slices.append(np.s_[:, (image_width - window_width):image_width])
        # Reverse slices to account for RTL
        slices.reverse()
        return resized_im, slices

    def infer_probability_vector(self, window: np.ndarray) -> np.ndarray:
        # TODO!
        return np.random.rand(self.num_classes)

    def infer_probability_matrix(self, image):
        resized, slices = self.resize_and_slice(image)
        mat = np.zeros((len(slices), self.num_classes))
        for i, slice in enumerate(slices):
            mat[i, :] = self.infer_probability_vector(resized[slice])
        return mat

    def classify_all(self):
        return [self.infer_probability_matrix(image) for image in tqdm(self.word_images, desc='Encoding images')]
