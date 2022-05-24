import os
import random

import albumentations as A
from PIL import Image, ImageOps
import cv2
import numpy as np
from tqdm import tqdm
from skimage.morphology import skeletonize, thin
from skimage.util import invert
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

# region GLOBAL VARIABLES - assuming original data folder is in some
PARENT_DIR = 'hwr-2022-dss-data-main\characters'
CROP_PARENT_DIR = 'Cropped_Images'
AUGMENTED_PARENT_DIR = 'Augmented_Characters'
THRESHOLD_PARENT_DIR = "Threshold_images"
FINAL_PARENT_DIR = "FINAL_IMAGES"
SAVE_EXTENSION = ".jpg"

PADDING = 2  # Number of Pixels of padding to add after Bounding Box
AUGMENTATIONS = 6


# endregion

def image_transforms(image, max_height, max_width):
    # TODO: REMOVE BRIGHTNESS AND CONTRAST
    # TODO: REMOVE BLURRING
    # TODO: remove sharpening

    # TODO: add dilation
    # TODO: Salt and Pepper Noise

    # TODO: apply all steps 5 times to each image
    # TODO: Upload

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

    """"
    # Image Resizing
    max_height = max_height
    max_width = max_width
    image_height = image.shape[0]
    image_width = image.shape[1]

    # height_ratio = (image_height/max_height)
    # width_ratio = (image_width/max_width)
    height_ratio = (max_height / image_height)
    width_ratio = (max_width / image_width)

    delta_height = 0
    delta_width = 0

    scale_value = height_ratio if height_ratio < width_ratio else width_ratio
    print(scale_value)

    if height_ratio < width_ratio:
        height = int(max_height)
        width = int(image_width * scale_value)
        delta_width = max_width - width
    else:
        height = int(image_height * scale_value)
        width = int(max_width)
        delta_height = max_height - height

    image = cv2.resize(image, (width, height), cv2.INTER_AREA)

    # Image Padding
    if delta_width == 0:
        # Padding on top / bottom
        top, bottom = delta_height // 2, delta_height // 2
        left, right = 0, 0
    else:
        # Padding on left / right
        top, bottom = 0, 0
        left, right = delta_width // 2, delta_width // 2

    image = cv2.copyMakeBorder(image,
                               top=top, bottom=bottom,
                               left=left, right=right,
                               value=[0, 0, 0],
                               borderType=cv2.BORDER_CONSTANT)
    print(image.shape)
    """
    # endregion

    # region Image Augmentations
    # Image Augmentations using Albumentations
    transform1 = A.Compose(
        [A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.25, p=1.0)]
    )

    transform2 = A.Compose(
        [A.Rotate(limit=(-15, 15), p=1.0),
         A.Blur(blur_limit=(1, 2), p=1.0)]
    )

    transform3 = A.Compose(
        [A.Sharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0), p=1)]
    )

    # Threshold to get rid of noise caused by cropping
    _, image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

    # Apply Albumentations
    random_brightness_contrast = transform1(image=image)["image"]
    rotation_and_blur = transform2(image=image)["image"]
    sharpening = transform3(image=image)["image"]

    # Partial Thinning of Images
    inverted_2D_image = _image_to_array(image, image.shape[0], image.shape[1])
    partially_thinned_image = invert(thin(image=inverted_2D_image, max_num_iter=3))
    partially_thinned_image = img_as_ubyte(partially_thinned_image)

    # Image Erosion
    kernel = np.ones((3, 3))
    eroded_image = invert(cv2.erode(src=invert(image),
                                    kernel=kernel,
                                    iterations=2))

    # region Grid View of Additional Albumentations
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original')
    ax[0].axis('off')
    ax[1].imshow(partially_thinned_image, cmap=plt.cm.gray)
    ax[1].set_title('Partially thinned')
    ax[1].axis('off')
    ax[2].imshow(inverted_2D_image, cmap=plt.cm.gray)
    ax[2].set_title('Inverted')
    ax[2].axis('off')
    ax[3].imshow(eroded_image, cmap=plt.cm.gray)
    ax[3].set_title('Eroded')
    ax[3].axis('off')

    fig.tight_layout()
    plt.show()
    """
    # endregion

    # endregion

    return image, random_brightness_contrast, rotation_and_blur, sharpening, partially_thinned_image, eroded_image


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

            tuple = image_transforms(image, max_height=max_height, max_width=max_width)

            for i in range(len(tuple)):
                image_name = character_name + "_" + str(ENUMERATOR) + "_" + str(i) + SAVE_EXTENSION
                new_image_path = os.path.join(new_path, image_name)
                cv2.imwrite(new_image_path, tuple[i])

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
        for j in range(0, len(os.listdir(file)), AUGMENTATIONS):
            # One Letter
            tmp = []
            for k in range(AUGMENTATIONS):
                # One set of augments
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


if __name__ == '__main__':
    print("[INFO] Performing Data Augmentation\n===========================================")
    # threshold()
    # crop_images()
    max_height, max_width = determine_largest_size()
    augment(max_height=max_height, max_width=max_width)
    train_test_validation(use_augmented=False,
                          no_validation=False,
                          train_percent=0.8,
                          test_percent=0.1,
                          validation_percent=0.1,
                          parent_dir="FINAL_IMAGES")
