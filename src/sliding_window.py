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

        if new_width < window_width:
            if len(image.shape) == 3:
                new_resized = np.zeros((*self.window_size, 3), dtype=np.uint8)
            else:
                new_resized = np.zeros(self.window_size, dtype=np.uint8)
            offset = (window_width - new_width) // 2
            new_resized[:, offset:offset+new_width, ...] = resized_im
            resized_im = new_resized
            new_width = window_width
            image_width = window_width

        # print('jochie', image_width, window_width, self.hop_size, resized_im.shape, ratio, new_width, image.shape)
        num_steps, rem = divmod((image_width - window_width), self.hop_size)
        slices = []
        if num_steps-rem > 0:
            for i in range(0, num_steps-rem):
                offset = i * self.hop_size
                # print('resi', offset, window_width)
                slices.append(np.s_[:, offset:offset + window_width, ...])
            for i in range(num_steps-rem, num_steps):
                offset = (i * self.hop_size) + 1
                # print('resi2', offset, window_width, i, num_steps, rem)
                slices.append(np.s_[:, offset:offset + window_width, ...])
        else:
            slices.append(np.s_[:, 0:window_width, ...])
            if rem > 0:
                slices.append(np.s_[:, rem:window_width+rem, ...])
        # Reverse slices to account for RTL
        if self.right_to_left:
            slices.reverse()
        return resized_im, slices

    def infer_probability_vector(self, window: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.random.rand(self.num_classes)
        # print('infer vec', window.shape)
        if len(window.shape) == 3: # (71, 40, 3) --> (71, 40)
            window = cv.cvtColor(window, cv.COLOR_RGB2GRAY)
        window = np.expand_dims(window, axis=(0, 3))
        # print(window.shape)
        window = np.repeat(window, self.channels, axis=3)
        # print('infer vec2', window.shape)
        assert window.shape == (1, 71, 40, self.channels)
        predictions = self.model.predict(window)[0]
        # print('Predictions shape:', predictions.shape, ' Predictions: ', predictions)
        return predictions

    def infer_probability_matrix(self, image):
        resized, slices = self.resize_and_slice(image)
        mat = np.zeros((len(slices), self.num_classes))
        for i, slice in enumerate(slices):
            # print('infer mat', slice)
            mat[i, :] = self.infer_probability_vector(resized[slice])
        return mat

    def classify_all(self):
        return [self.infer_probability_matrix(image) for image in tqdm(self.word_images, desc='Encoding images')]
