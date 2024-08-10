import argparse
import sys
import cv2 as cv
import numpy as np


def read_image(path):
    try:
        img = cv.imread(path)
    except:
        sys.exit("Could not find image! Please recheck path")
    return img


def rescaler(frame, scale=0.75, dimension=None):
    if not dimension:
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dim = (width, height)
    else:
        dim = dimension
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)


def display_image(img, scale=0.5, save_path=None):
    # rescale image
    img = rescaler(img, scale=scale, dimension=(1000, 600))
    cv.imshow("Display Window", img)
    k = cv.waitKey(0)
    if save_path:
        if k == ord("s"):
            cv.imwrite(f"{save_path}/image.jpg", img)
    return True


# ================Color Space Conversion==============#
def bgr_to_gray(img):
    new_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return new_image


def bgr_to_hsv(img):
    new_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    return new_image


def bgr_to_lab(img):
    new_image = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    return new_image


def bgr_to_xyz(img):
    new_image = cv.cvtColor(img, cv.COLOR_BGR2XYZ)
    return new_image


def bgr_to_ycrcb(img):
    new_image = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    return new_image


def stack_images(images, direction, convert=True):
    image_1 = images.get('image_1')
    image_2 = images.get('image_2')

    if direction == 'horizontal':
        if convert:
            image_1 = cv.cvtColor(image_1, cv.COLOR_GRAY2BGR)
            image_2 = cv.cvtColor(image_2, cv.COLOR_GRAY2BGR)
        else:
            image_2 = cv.cvtColor(image_2, cv.COLOR_GRAY2BGR)
        stack = np.hstack((image_1, image_2))
    else:
        stack = np.vstack((image_1, image_2))

    return stack


if __name__ == '__main__':
    args = sys.argv

    color_space = args[1].lower()
    image_path = args[2].lower()

    image = read_image(image_path)

    if color_space == "-xyz":
        converted_image = bgr_to_xyz(image)
    elif color_space == "-lab":
        converted_image = bgr_to_lab(image)
    elif color_space == "-ycrcb":
        converted_image = bgr_to_ycrcb(image)
    elif color_space == "-hsb":
        converted_image = bgr_to_hsv(image)
    else:
        raise ValueError(f"Color space '{color_space}' is not supported.")
    a_component, b_component, c_component = cv.split(converted_image)

    stack_1 = stack_images(
        images={
            'image_1': image,
            'image_2': a_component,
        },
        direction='horizontal',
        convert=False
    )

    stack_2 = stack_images(
        images={
            'image_1': b_component,
            'image_2': c_component,
        },
        direction='horizontal'
    )

    # Combine the two rows into a single image
    combined_image = stack_images(
        images={
            'image_1': stack_1,
            'image_2': stack_2,
        },
        direction='vertical',
    )

    display_image(combined_image)
