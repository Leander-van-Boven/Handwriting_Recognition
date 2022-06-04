import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.iam.model import num_images, WINDOW_SIZE

FINAL_PARENT_DIR = Path('../../data/iam/IMData_Split').resolve()
SAVE_EXTENSION = '.jpg'
HOP_SIZE = WINDOW_SIZE // 4


def create_blank_images():
    # Load n-grams for probabilities
    with open(Path('../../data/iam/ngrams/ngrams_processed.json').resolve(), 'r') as f:
        ngrams = json.load(f)
        uni_grams = ngrams['uni_grams']
        bi_grams = ngrams['bi_grams']

    # Generate 'Blank' character images (i.e. sliding window location 'between' two characters)
    # for train, validation and test
    for ds_type, num_blank in zip(['Train', 'Validation', 'Test'], num_images):
        path = FINAL_PARENT_DIR / ds_type
        blank_path = path / 'ZBlank'
        if not blank_path.exists():
            os.mkdir(blank_path)

        for i in tqdm(range(num_blank // 3), desc=f'Generating Blank images for {ds_type}'):
            char_one = np.random.choice([key for key in uni_grams.keys()], p=[value for value in uni_grams.values()])
            # char_one = None
            # while char_one is None:
            #     char_one = np.random.choice([key for key in uni_grams.keys()], p=[value for value in uni_grams.values()])
            #     if char_one in ['Kaf-final', 'Nun-final', 'Tsadi-final']:
            #         char_one = None
            char_two = np.random.choice([key for key in bi_grams[char_one].keys()],
                                        p=[value for value in bi_grams[char_one].values()])

            # convert string character to hex
            char_one = hex(ord(char_one))[2:]
            char_two = hex(ord(char_two))[2:]

            char_one_file = random.choice(os.listdir(path / char_one))
            char_two_file = random.choice(os.listdir(path / char_two))

            char_one_img = cv2.imread(str(path / char_one / char_one_file))
            char_two_img = cv2.imread(str(path / char_two / char_two_file))

            padding = (char_one_img.shape[0] - WINDOW_SIZE) // 2
            merged_img = np.concatenate((char_one_img[:, padding:padding+WINDOW_SIZE],
                                         char_two_img[:, padding:padding+WINDOW_SIZE]), axis=1)

            for j in range(-1, 2):
                hop_compensation = j * (HOP_SIZE // 4)
                blank_img = merged_img[:, 2 * HOP_SIZE + hop_compensation:2 * HOP_SIZE + WINDOW_SIZE + hop_compensation]
                blank_img = cv2.copyMakeBorder(blank_img, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                cv2.imwrite(str(blank_path / f'Blank_{ds_type.lower()}_{i * 3 + j}{SAVE_EXTENSION}'), blank_img)


if __name__ == '__main__':
    print("[INFO] Performing Data Augmentation\n===========================================")
    create_blank_images()
