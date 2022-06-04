import cv2 as cv
import numpy as np
from tqdm import tqdm


class SlidingWindowClassifier:
    def __init__(self, model, num_classes, word_images, right_to_left, conf):
        self.model = model
        self.num_classes = num_classes
        self.word_images = word_images
        self.window_size = conf.window.size
        self.hop_size = conf.window.hop_size
        self.right_to_left = right_to_left
        self.channels = conf.window.channels

    def resize_and_slice(self, image):
        assert self.hop_size > 0

        ratio = self.window_size[0] / image.shape[0]
        new_width = int(image.shape[1] * ratio)
        resized_im = cv.resize(image, (new_width, self.window_size[0]))

        image_width = resized_im.shape[1]
        window_width = self.window_size[1]
        num_steps, rem = divmod((image_width - window_width), self.hop_size)
        slices = []
        for i in range(0, num_steps-rem):
            offset = i * self.hop_size
            slices.append(np.s_[:, offset:offset + window_width, ...])
        for i in range(num_steps-rem, num_steps):
            offset = (i + 1) * self.hop_size
            slices.append(np.s_[:, offset:offset + window_width, ...])
            assert offset+window_width <= image_width
        # Reverse slices to account for RTL
        if self.right_to_left:
            slices.reverse()
        return resized_im, slices

    def infer_probability_vector(self, window: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.random.rand(self.num_classes)
        window = np.expand_dims(window, axis=(0,3))
        window = np.repeat(window, self.channels, axis=3)
        assert window.shape == (1, 71, 40, self.channels)
        predictions = self.model.predict(window)[0]
        # print('Predictions shape:', predictions.shape, ' Predictions: ', predictions)
        return predictions

    def infer_probability_matrix(self, image):
        resized, slices = self.resize_and_slice(image)
        mat = np.zeros((len(slices), self.num_classes))
        for i, slice in enumerate(slices):
            mat[i, :] = self.infer_probability_vector(resized[slice])
        return mat

    def classify_all(self):
        return [self.infer_probability_matrix(image) for image in tqdm(self.word_images, desc='Encoding images')]
