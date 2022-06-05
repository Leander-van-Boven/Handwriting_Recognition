import json
import os
import random
from pathlib import Path

import albumentations as A
from PIL import Image, ImageOps
import cv2
import numpy as np
from tqdm import tqdm
from skimage.morphology import skeletonize, thin
from skimage.util import invert
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

from src.dss.model import num_images


# region GLOBAL VARIABLES - assuming original data folder is in some
PARENT_DIR = Path('../../data/dss/characters').resolve()
CROP_PARENT_DIR = 'Cropped_Images'
AUGMENTED_PARENT_DIR = 'Augmented_Characters'
THRESHOLD_PARENT_DIR = "Threshold_images"
FINAL_PARENT_DIR = Path('../../data/dss/FINAL_IMAGES_AUGMENTS').resolve()
SAVE_EXTENSION = ".jpg"

PADDING = 2  # Number of Pixels of padding to add after Bounding Box
AUGMENTATIONS = 5
NUM_SPECIAL_AUGMENTS = 5
WINDOW_SIZE = 40
HOP_SIZE = WINDOW_SIZE // 4
# endregion


def image_transforms(image, max_height, max_width, number_of_augments):
    # Pick Largest Size
    max_height = max_height if max_height > max_width else max_width
    max_width = max_width if max_width > max_height else max_height

    # region Image Resizing Code
    image_height = image.shape[0]
    image_width = image.shape[1]

    delta_height = max_height - image_height
    delta_width = max_width - image_width

    top, bottom = delta_height // 2, delta_height // 2
    left, right = delta_width // 2, delta_width // 2

    while top + bottom + image_height != max_height:
        tmp = random.randrange(0, 1)
        if tmp == 0:
            top += 1
        else:
            bottom += 1

    while left + right + image_width != max_width:
        tmp = random.randrange(0, 1)
        if tmp == 0:
            left += 1
        else:
            right += 1

    image = cv2.copyMakeBorder(image,
                               top=top, bottom=bottom,
                               left=left, right=right,
                               value=[255, 255, 255],
                               borderType=cv2.BORDER_CONSTANT)

    # endregion

    # region Image Augmentations
    # Image Augmentations using Albumentations
    transform2 = A.Compose(
        [A.Rotate(limit=(-15, 15), p=1.0)]
    )

    # Threshold to get rid of noise caused by cropping
    _, image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

    # Apply Albumentations
    rotations = []
    for i in range(number_of_augments):
        rotations.append(transform2(image=image)["image"])

    rotation = transform2(image=image)["image"]

    # Partial Thinning of Images
    inverted_2D_image = _image_to_array(image, image.shape[0], image.shape[1])

    partially_thinned_image = invert(thin(image=inverted_2D_image, max_num_iter=3))
    partially_thinned_image = img_as_ubyte(partially_thinned_image)

    partially_thinned_images = []
    for i in range(number_of_augments):
        partially_thinned_image = invert(thin(image=inverted_2D_image, max_num_iter=3))
        partially_thinned_image = img_as_ubyte(partially_thinned_image)
        partially_thinned_images.append(partially_thinned_image)

    # Salt and Pepper Noise
    salt_and_pepper_noise_images = []
    for i in range(number_of_augments):
        salt_and_pepper_noise_images.append(_salt_and_pepper_noise(image, image.shape[0], image.shape[1]))
    salt_and_pepper_image = _salt_and_pepper_noise(image, image.shape[0], image.shape[1])

    # Image Erosion
    kernel = np.ones((3, 3))
    eroded_image = invert(cv2.erode(src=invert(image),
                                    kernel=kernel,
                                    iterations=2))

    eroded_images = []
    for i in range(number_of_augments):
        eroded_images.append(invert(cv2.erode(src=invert(image),
                                    kernel=kernel,
                                    iterations=2)))

    # Image Dilation
    dilated_im = invert(cv2.dilate(src=invert(image),
                            kernel=kernel,
                            iterations=2))

    dilated_images = []
    for i in range(number_of_augments):
        dilated_images.append(invert(cv2.dilate(src=invert(image),
                            kernel=kernel,
                            iterations=2)))

    # region Grid View of Additional Albumentations
    '''
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original')
    ax[0].axis('off')
    ax[1].imshow(eroded_image, cmap=plt.cm.gray)
    ax[1].set_title('erosion')
    ax[1].axis('off')
    ax[2].imshow(dilated_im, cmap=plt.cm.gray)
    ax[2].set_title('dilated')
    ax[2].axis('off')
    ax[3].imshow(salt_and_pepper_image, cmap=plt.cm.gray)
    ax[3].set_title('salt and pepper')
    ax[3].axis('off')

    fig.tight_layout()
    plt.show()
    '''
    # endregion

    # endregion

    # Convert all to tuples for easy unpacking
    rotations = tuple(rotations)
    partially_thinned_images = tuple(partially_thinned_images)
    salt_and_pepper_noise_images = tuple(salt_and_pepper_noise_images)
    eroded_images = tuple(eroded_images)
    dilated_images = tuple(dilated_images)

    # return image, rotation, salt_and_pepper_image, dilated_im, partially_thinned_image, eroded_image
    return image, rotations, partially_thinned_images, salt_and_pepper_noise_images, eroded_images, dilated_images


def _image_to_array(image, height, width):
    new_im = []

    for i in range(height):
        new_img_row = []
        for j in range(width):
            # Invert Thresholded Image for Skeletonization
            val = 255 if min(image[i][j]) < 50 else 0  # Threshold
            new_img_row.append(val)

        new_im.append(new_img_row)

    return new_im


def _salt_and_pepper_noise(image_arr, height, width):
    new_im = image_arr.copy()
    num_pixels_black = random.randint(100, int((height*width)/15)) # 5041
    num_pixels_white = random.randint(100, int((height*width)/15)) # 5041

    for i in range(num_pixels_black):
        y_cord = random.randint(0, height - 1)
        x_cord = random.randint(0, width - 1)

        new_im[y_cord][x_cord] = 0

    for i in range(num_pixels_white):
        y_cord = random.randint(0, height - 1)
        x_cord = random.randint(0, width - 1)

        new_im[y_cord][x_cord] = 255

    return new_im


def augment(max_height, max_width):
    try:
        os.mkdir(AUGMENTED_PARENT_DIR)
    except OSError as error:
        pass

    for filename in tqdm(os.listdir(CROP_PARENT_DIR), desc='Applying Augmentations to Image Files'):
        character_name = filename
        f = os.path.join(CROP_PARENT_DIR, filename)
        new_path = os.path.join(AUGMENTED_PARENT_DIR, filename)
        try:
            os.mkdir(new_path)
        except OSError as error:
            pass

        ENUMERATOR = 0
        for character_data in os.listdir(f):
            # Image Files
            file = os.path.join(f, character_data)

            image = cv2.imread(file)

            tuple = image_transforms(image, max_height=max_height, max_width=max_width, number_of_augments=NUM_SPECIAL_AUGMENTS)
            # image, then arrays
            for i in range(len(tuple)):
                if i == 0:
                    image_name = character_name + "_" + str(ENUMERATOR) + "_" + str(i) + SAVE_EXTENSION
                    new_image_path = os.path.join(new_path, image_name)
                    cv2.imwrite(new_image_path, tuple[i])
                else:
                    for j in range(NUM_SPECIAL_AUGMENTS):
                        image_name = character_name + "_" + str(ENUMERATOR) + "_" + str(i) + "_" + str(j) + SAVE_EXTENSION
                        new_image_path = os.path.join(new_path, image_name)
                        cv2.imwrite(new_image_path, tuple[i][j])

            ENUMERATOR += 1


def crop_images():
    try:
        os.mkdir(CROP_PARENT_DIR)
    except OSError as error:
        pass

    for filename in tqdm(os.listdir(THRESHOLD_PARENT_DIR), desc='Cropping Image Files'):
        character_name = filename
        f = os.path.join(THRESHOLD_PARENT_DIR, filename)
        new_path = os.path.join(CROP_PARENT_DIR, filename)

        try:
            os.mkdir(new_path)
        except OSError as error:
            pass

        for character_data in os.listdir(f):
            # Image Files
            file = os.path.join(f, character_data)
            character_data = character_data.split('.')[0]

            # region PIL Bounding Box
            image = Image.open(file)
            image.load()

            image = ImageOps.invert(image)

            image_box = image.getbbox()

            cropped_image = image.crop(image_box)
            cropped_image = ImageOps.invert(cropped_image)
            # endregion

            # region Advanced Bounding Box
            cropped_image_CV = np.asarray(cropped_image)
            original = cropped_image_CV.copy()

            gray = cv2.cvtColor(cropped_image_CV, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            ROI_num = 0
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]

            if not len(contours) == 0:
                # Calculate Largest ROI Area
                largest_ROI = 0
                largest_area = 0
                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)

                    area = w * h

                    if area > largest_area:
                        largest_ROI = c
                        largest_area = area

                x, y, w, h = cv2.boundingRect(largest_ROI)
                ROI_image = original[y:y + h, x:x + w]

                # Add Padding
                ROI_image = cv2.copyMakeBorder(ROI_image,
                                               top=PADDING, bottom=PADDING,
                                               left=PADDING, right=PADDING,
                                               value=[255, 255, 255],
                                               borderType=cv2.BORDER_CONSTANT)

                image_name = character_data + SAVE_EXTENSION
                new_image_path = os.path.join(new_path, image_name)

                cv2.imwrite(new_image_path, ROI_image)
            else:
                # Add Padding
                cropped_image_CV = cv2.copyMakeBorder(cropped_image_CV,
                                                      top=PADDING, bottom=PADDING,
                                                      left=PADDING, right=PADDING,
                                                      value=[255, 255, 255],
                                                      borderType=cv2.BORDER_CONSTANT)

                image_name = character_data + SAVE_EXTENSION
                new_image_path = os.path.join(new_path, image_name)

                cv2.imwrite(new_image_path, cropped_image_CV)
            # endregion


def determine_largest_size():
    max_image_height = 0
    max_image_width = 0

    max_image_height_name = ""
    max_image_width_name = ""

    for filename in os.listdir(CROP_PARENT_DIR):
        f = os.path.join(CROP_PARENT_DIR, filename)

        for character_data in os.listdir(f):
            # Image Files
            file = os.path.join(f, character_data)

            image = cv2.imread(file)
            try:
                if max_image_height < image.shape[0]:
                    max_image_height_name = file
                if max_image_width < image.shape[1]:
                    max_image_width_name = file
            except AttributeError as error:
                print(file)
                exit()

            max_image_height = max_image_height if max_image_height > image.shape[0] else image.shape[0]
            max_image_width = max_image_width if max_image_width > image.shape[1] else image.shape[1]

    print("Max Image Height: %i" % max_image_height)
    print("Max Image Height File Name: %s" % max_image_height_name)
    print("Max Image Width: %i" % max_image_width)
    print("Max Image Width File Name: %s" % max_image_width_name)

    return max_image_height, max_image_width


def threshold():
    try:
        os.mkdir(THRESHOLD_PARENT_DIR)
    except OSError as error:
        pass

    for filename in tqdm(os.listdir(PARENT_DIR), desc='Applying Thresholding to Image Files'):
        f = os.path.join(PARENT_DIR, filename)
        new_path = os.path.join(THRESHOLD_PARENT_DIR, filename)

        try:
            os.mkdir(new_path)
        except OSError as error:
            pass

        for character_data in os.listdir(f):
            # Image Files
            file = os.path.join(f, character_data)

            image = cv2.imread(file)

            _, image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

            file = file.split('.')[0]
            image_name = file.split('\\')[len(file.split('\\')) - 1] + SAVE_EXTENSION

            new_image_path = os.path.join(new_path, image_name)

            cv2.imwrite(new_image_path, image)


def train_test_validation(use_augmented=False, no_validation=False, train_percent=0.8, test_percent=0.1,
                          validation_percent=0.1, parent_dir=FINAL_PARENT_DIR):
    train = []
    test = []
    validation = []

    # 80 | 10 | 10 split
    original_images = []
    edited_images = []

    for i in tqdm(range(len(os.listdir(AUGMENTED_PARENT_DIR))), desc="Reading In Edited Images"):
        # All letters
        file = os.path.join(AUGMENTED_PARENT_DIR, os.listdir(AUGMENTED_PARENT_DIR)[i])
        letter = []
        augments_per_image = AUGMENTATIONS * NUM_SPECIAL_AUGMENTS + 1  # 1 is original image, rest are augments
        # for j in range(0, len(os.listdir(file)), AUGMENTATIONS):
        for j in range(0, len(os.listdir(file)), augments_per_image):
            # One Letter
            tmp = []
            # for k in range(AUGMENTATIONS):
            for k in range(augments_per_image):

                image_path = os.path.join(file, os.listdir(file)[j + k])
                image = cv2.imread(image_path)
                tmp.append(image)
            letter.append(tmp)
        edited_images.append(letter)

    print("[INFO] Read In Edited Images")

    for i in tqdm(range(len(edited_images)), desc="Train/Test/Validation Generation"):
        letter_name = os.listdir(AUGMENTED_PARENT_DIR)[i]

        try:
            os.mkdir(parent_dir)
            os.mkdir(os.path.join(parent_dir, "Train"))
            os.mkdir(os.path.join(parent_dir, "Test"))
            if not no_validation:
                os.mkdir(os.path.join(parent_dir, "Validation"))
        except OSError as error:
            pass

        try:
            os.mkdir(os.path.join(parent_dir, "Train", letter_name))
            os.mkdir(os.path.join(parent_dir, "Test", letter_name))
            if not no_validation:
                os.mkdir(os.path.join(parent_dir, "Validation", letter_name))
        except OSError as error:
            pass

        train_amount = int(len(edited_images[i]) * train_percent)
        test_amount = int(len(edited_images[i]) * test_percent)
        validation_amount = int(len(edited_images[i]) * validation_percent)

        if len(edited_images[i]) != train_amount + test_amount + validation_amount:
            # print("[ERROR] Horrible Things Will Occur")
            # print("Train Amount: %i" % train_amount)
            # print("Test Amount: %i" % test_amount)
            # print("Validation Amount: %i" % validation_amount)
            # print("Total Amount: %i" % len(edited_images[i]))

            remaining = len(edited_images[i]) - (train_amount + test_amount + validation_amount)
            train_amount += remaining

            # print("Train Amount: %i" % train_amount)
            # print("Test Amount: %i" % test_amount)
            # print("Validation Amount: %i" % validation_amount)
            # print("Total Amount: %i" % len(edited_images[i]))

        # Add To The Sets
        train_subset = []
        test_subset = []
        validation_subset = []

        # Only take first image for test and validation because that is only cropped and thresholded
        if not use_augmented:
            for j in range(test_amount):
                tmp = random.randrange(0, len(edited_images[i]))
                test_subset.append(edited_images[i][tmp][0])
                del edited_images[i][tmp]

            for j in range(validation_amount):
                tmp = random.randrange(0, len(edited_images[i]))
                if not no_validation:
                    validation_subset.append(edited_images[i][tmp][0])
                else:
                    test_subset.append(edited_images[i][tmp][0])
                del edited_images[i][tmp]
        else:
            for j in range(test_amount):
                tmp = random.randrange(0, len(edited_images[i]))
                for im in edited_images[i][tmp]:
                    test_subset.append(im)
                del edited_images[i][tmp]

            for j in range(validation_amount):
                tmp = random.randrange(0, len(edited_images[i]))
                for im in edited_images[i][tmp]:
                    if not no_validation:
                        validation_subset.append(im)
                    else:
                        test_subset.append(im)
                del edited_images[i][tmp]

        for j in edited_images[i]:
            for k in j:
                train_subset.append(k)

        # Image Names Generation
        TEST = "test"
        TRAIN = "train"
        VALIDATION = "validation"

        ENUMERATOR = 0
        for img in test_subset:
            file_name = letter_name + "_" + TEST + "_" + str(ENUMERATOR) + SAVE_EXTENSION
            path = os.path.join(parent_dir, "Test", letter_name, file_name)
            cv2.imwrite(path, img)
            ENUMERATOR += 1

        if not no_validation:
            ENUMERATOR = 0
            for img in validation_subset:
                file_name = letter_name + "_" + VALIDATION + "_" + str(ENUMERATOR) + SAVE_EXTENSION
                path = os.path.join(parent_dir, "Validation", letter_name, file_name)
                cv2.imwrite(path, img)
                ENUMERATOR += 1

        ENUMERATOR = 0
        for img in train_subset:
            file_name = letter_name + "_" + TRAIN + "_" + str(ENUMERATOR) + SAVE_EXTENSION
            path = os.path.join(parent_dir, "Train", letter_name, file_name)
            cv2.imwrite(path, img)
            ENUMERATOR += 1


def create_blank_images():
    # Load n-grams for probabilities
    with open(Path('../../data/dss/ngrams/ngrams_processed.json').resolve(), 'r') as f:
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
            char_one = None
            while char_one is None:
                char_one = np.random.choice([key for key in uni_grams.keys()], p=[value for value in uni_grams.values()])
                if char_one in ['Kaf-final', 'Nun-final', 'Tsadi-final']:
                    char_one = None
            char_two = np.random.choice([key for key in bi_grams[char_one].keys()],
                                        p=[value for value in bi_grams[char_one].values()])

            char_one_file = random.choice(os.listdir(path / char_one))
            char_two_file = random.choice(os.listdir(path / char_two))

            char_one_img = cv2.imread(str(path / char_one / char_one_file))
            char_two_img = cv2.imread(str(path / char_two / char_two_file))

            merged_img = np.concatenate((char_one_img[:, 15:55], char_two_img[:, 15:55]), axis=1)

            for j in range(1, 4):
                blank_img = merged_img[:, j * HOP_SIZE:j * HOP_SIZE + WINDOW_SIZE]
                blank_img = cv2.copyMakeBorder(blank_img, 0, 0, 15, 16, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                cv2.imwrite(str(blank_path / f'Blank_{ds_type.lower()}_{i * 3 + j}{SAVE_EXTENSION}'), blank_img)


if __name__ == '__main__':
    print("[INFO] Performing Data Augmentation\n===========================================")
    threshold()
    crop_images()
    max_height, max_width = determine_largest_size()
    augment(max_height=max_height, max_width=max_width)
    train_test_validation(use_augmented=True,
                          no_validation=False,
                          train_percent=0.8,
                          test_percent=0.1,
                          validation_percent=0.1)
    create_blank_images()
