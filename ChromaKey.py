import argparse
import sys
import cv2 as cv
import numpy as np


def read_image(path):
    try:
        img = cv.imread(path, cv.IMREAD_UNCHANGED)
    except:
        sys.exit("Could not find image! Please recheck path")
    return img


def get_blank_image(color, size=(500, 500, 3)):
    if color == 'black':
        color_code = 0
    else:
        color_code = 255
    blank = np.full(size, color_code, dtype="uint8")
    return blank


def get_scale(original_width, original_height, min_width, max_width, min_height, max_height):
    aspect_ratio = original_width / original_height

    # Calculate potential new dimensions
    width_based_height = min_width / aspect_ratio
    height_based_width = min_height * aspect_ratio

    # Decide based on constraints
    if min_width <= max_width and min_height <= width_based_height <= max_height:
        new_width = min_width
        new_height = width_based_height
    elif min_height <= max_height and min_width <= height_based_width <= max_width:
        new_width = height_based_width
        new_height = min_height
    else:
        # Fallback, if no conditions match, just use the max width and height
        new_width = max_width
        new_height = int(max_width / aspect_ratio)
        if new_height > max_height:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)

    return new_width, new_height


def rescaler(frame, scaling_prop, keep_aspect=True):
    image_shape = frame.shape[:2]
    dimension = scaling_prop.get('dimension', None)
    scale = scaling_prop.get('scale', None)
    if keep_aspect:
        min_max_width = scaling_prop.get('min_max_width', None)
        min_max_height = scaling_prop.get('min_max_width', None)
        dim = get_scale(
            image_shape[1],
            image_shape[0],
            min_max_width[0],
            min_max_width[1],
            min_max_height[0],
            min_max_height[1]
        )

    else:
        if not dimension:
            width = int(frame.shape[1] * scale)
            height = int(frame.shape[0] * scale)
            dim = (width, height)
        else:
            dim = dimension
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)


def display_image(img, rescale=False, save_path=None):
    if rescale:
        img = rescaler(
            img,
            scaling_prop={
                'min_max_width': (1200, 1280),
                'min_max_height': (900, 960),
            },
            keep_aspect=True
        )

    img_shape = img.shape[:2]

    cv.imshow(f"Display Window - Aspect ratio: {img_shape[1] / img_shape[0]}", img)
    k = cv.waitKey(0)
    if save_path:
        if k == ord("s"):
            cv.imwrite(f"{save_path}/image.jpg", img)
    cv.destroyAllWindows()
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


def stack_images(images, direction, convert=True, generic=False):
    image_1 = images.get('image_1')
    image_2 = images.get('image_2')

    if direction == 'horizontal':
        if not generic:
            stack = np.hstack((image_1, image_2))
        else:
            if convert:
                image_1 = cv.cvtColor(image_1, cv.COLOR_GRAY2BGR)
                image_2 = cv.cvtColor(image_2, cv.COLOR_GRAY2BGR)
            else:
                image_2 = cv.cvtColor(image_2, cv.COLOR_GRAY2BGR)
            stack = np.hstack((image_1, image_2))
    else:
        stack = np.vstack((image_1, image_2))
    return stack


def determine_task(args_options):
    color_space_options = ['-XYZ', '-Lab']
    if args_options[1] in color_space_options:
        task = 'task_1'
    else:
        task = 'task_2'
    return task


def twin_image_resize(base_image, image_to_resize):
    base_image_shape = base_image.shape[:2]
    scenic_resized = rescaler(
        image_to_resize,
        scaling_prop={
            'dimension': (base_image_shape[1], base_image_shape[0]),
        },
        keep_aspect=False
    )
    return scenic_resized


def get_bg_threshold(space):
    if space == 'hsv':
        lower_green_range = np.array([35, 50, 50])
        upper_green_range = np.array([85, 255, 255])
    else:
        lower_green_range = ...
        upper_green_range = ...
    return lower_green_range, upper_green_range


def mask_out(img, space='hsv'):
    # Convert image to hsv to ensure better color distinction
    img = bgr_to_hsv(img)
    # img = cv.morphologyEx(img, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # img = cv.morphologyEx(img, cv.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Get the green color threshold
    lower_thresh, upper_thresh = get_bg_threshold(space)

    # Create a mask that identifies the green regions
    img = cv.inRange(img, lower_thresh, upper_thresh)

    # Define a kernel size for morphological operation
    kernel = np.ones((3, 3), np.uint8)  # You can experiment with the kernel size

    # Apply morphological opening to remove small white dots
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    # Improve the mask using morphological operations

    return img


def get_offsets(bg_scene, fg_scene):
    h_1, w_1 = bg_scene.shape[:2]
    h_2, w_2 = fg_scene.shape[:2]

    hor_offset = (w_1 - w_2) // 2
    ver_offset = (h_1 - h_2) // 2
    return h_1, h_2, w_1, w_2, hor_offset, ver_offset


def task_1(col_space, img_path):
    image = read_image(img_path)
    if col_space == "-xyz":
        converted_image = bgr_to_xyz(image)
    elif col_space == "-lab":
        converted_image = bgr_to_lab(image)
    elif col_space == "-ycrcb":
        converted_image = bgr_to_ycrcb(image)
    elif col_space == "-hsb":
        converted_image = bgr_to_hsv(image)
    else:
        raise ValueError(f"Color space '{col_space}' is not supported.")
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

    return True


def task_2_(green_screen_path, scenic_path):
    green_image = read_image(green_screen_path)
    scenic_image = read_image(scenic_path)

    # Resize the green screen image to match the scenic background
    resized_green_screen = twin_image_resize(scenic_image, green_image)

    # Create a mask based on the objects on the screen using green color thresholds
    masked_out_image = mask_out(resized_green_screen, space='hsv')

    # invert the color on the mask to make the background black
    inverted_masked = cv.bitwise_not(masked_out_image)

    extracted_objects = cv.bitwise_and(resized_green_screen, resized_green_screen, mask=inverted_masked)

    h_1, h_2, w_1, w_2, hor_offset, ver_offset = get_offsets(scenic_image, extracted_objects)

    objects_region_of_interest = scenic_image[ver_offset: ver_offset + h_2, hor_offset: hor_offset + w_2]

    new_scenic = cv.bitwise_and(objects_region_of_interest, objects_region_of_interest, mask=masked_out_image)

    # bg = cv.bitwise_and(scenic_image, scenic_image, mask=masked_out_image)
    final = cv.add(new_scenic, extracted_objects)

    scenic_image[ver_offset: ver_offset + h_2, hor_offset: hor_offset + w_2] = final
    display_image(final)


    # stack = stack_images(images={
    #     'image_1': green_image,
    #     'image_2': final,
    #     },
    #     direction='horizontal',
    #     convert=False
    # )


    return True


def task_2(green_screen_path, scenic_path):
    green_image = read_image(green_screen_path)
    scenic_image = read_image(scenic_path)

    # Resize the green screen image to match the scenic background
    resized_green_image = twin_image_resize(scenic_image, green_image)

    # Mask the objects in the green area
    masked_out_green_image = mask_out(resized_green_image, space='hsv')

    inverted_mask = cv.bitwise_not(masked_out_green_image)

    extracted_objects = cv.bitwise_and(resized_green_image, resized_green_image, mask=inverted_mask)

    # Generate offsets
    h_1, h_2, w_1, w_2, hor_offset, ver_offset = get_offsets(scenic_image, extracted_objects)

    # Generate white background and place objects
    white_scenic = get_blank_image('white', resized_green_image.shape)
    white_objects_region_of_interest = white_scenic[ver_offset: ver_offset + h_2, hor_offset: hor_offset + w_2]
    new_white_scenic = cv.bitwise_and(white_objects_region_of_interest, white_objects_region_of_interest, mask=masked_out_green_image)
    final_white_scene_with_person = cv.add(new_white_scenic, extracted_objects)
    # white_scenic[ver_offset: ver_offset + h_2, hor_offset: hor_offset + w_2] = final_white_scene_with_person

    objects_region_of_interest = scenic_image[ver_offset: ver_offset + h_2, hor_offset: hor_offset + w_2]
    new_scenic = cv.bitwise_and(objects_region_of_interest, objects_region_of_interest, mask=masked_out_green_image)
    final_combined = cv.add(new_scenic, extracted_objects)

    # scenic_image[ver_offset: ver_offset + h_2, hor_offset: hor_offset + w_2] = final_combined

    stack_1 = stack_images(
        images={
            'image_1': resized_green_image,
            'image_2': final_white_scene_with_person,
        },
        direction='horizontal',
        convert=False
    )
    stack_2 = stack_images(
        images={
            'image_1': scenic_image,
            'image_2': final_combined,
        },
        direction='horizontal',
        convert=False
    )
    # Combine the two rows into a single image
    combined_image = stack_images(
        images={
            'image_1': stack_1,
            'image_2': stack_2,
        },
        direction='vertical',
    )
    display_image(combined_image, rescale=True)

    return True


if __name__ == '__main__':
    args = sys.argv
    task_to_do = determine_task(args)

    if task_to_do == 'task_1':
        color_space = args[1].lower()
        image_path = args[2].lower()
        task_1(color_space, image_path)
    else:
        scenic_image_path = args[1]
        green_image_path = args[2]

        task_2(green_image_path, scenic_image_path)
