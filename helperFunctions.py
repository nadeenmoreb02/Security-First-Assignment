import itertools
import cv2
import glob
from fingerprint_image_enhancer import FingerprintImageEnhancer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Function to read enhanced images
def read_enhanced_images():
    # Path to image directory
    images_path = str('data/fprdata/DB2_B')
    # Get list of image filenames
    file_names = [img for img in glob.glob(images_path + "/*.tif")]
    # Sort filenames
    file_names.sort()
    print(file_names)
    return file_names

# Function to extract image name from filename
def get_image_name(filename):
    image = filename.split('/')
    return image[len(image)-1]

# Function to extract user ID from filename
def get_user_id(filename):
    return get_image_name(filename).split('_')[1]

# Function to split dataset into template and testing sets
def split_dataset(data, test_size):
    template, test = train_test_split(data, test_size=test_size, random_state=42)
    return template, test

# Function for image enhancement using Gabor filterbank
def enhance_image(image):
    image_enhancer = FingerprintImageEnhancer()
    output = image_enhancer.enhance(image)

    # Ensure output is grayscale and uint8
    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY) if len(output.shape) == 3 else output
    output = output.astype(np.uint8) if output.dtype != np.uint8 else output

    # Normalize the image
    normalized_img = cv2.normalize(output, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return normalized_img

# Function to get genuine and impostor scores
def get_genuine_impostor_scores(all_scores, identical):
    genuine_scores = [score[1] for score, ident in zip(all_scores, identical) if ident == 1]
    impostor_scores = [score[1] for score, ident in zip(all_scores, identical) if ident != 1]
    return genuine_scores, impostor_scores

# Function to generate template and test data
def generate_template_and_test_data(file_names):
    def process_image(filename):
        img = cv2.imread(filename)
        gray_img = img if len(img.shape) <= 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = enhance_image(gray_img)
        label = get_image_name(filename)
        return label, img

    def key_func(filename):
        return get_user_id(filename)

    template_set = {}
    test_set = {}

    for temp_label, group in itertools.groupby(sorted(file_names, key=key_func), key=key_func):
        data = [process_image(filename) for filename in group]
        template, test = split_dataset(data, 0.2)
        template_set.update(template)
        test_set.update(test)

    return template_set, test_set

# Initialize SIFT detector
sift = cv2.SIFT_create(500)

# Function to split authentication dataset
def authentication_dataset_split(images_set):
    all_templates_descriptors = {}
    for image_name, image in images_set.items():
        kp, des = sift.detectAndCompute(image, None)
        all_templates_descriptors[image_name] = des

    authentication_databases = {}
    temp_list = {}
    user_id = get_user_id(list(all_templates_descriptors.keys())[0])  # Initial user name
    last_key = list(all_templates_descriptors.keys())[-1]

    for image_name, feature_descriptor in all_templates_descriptors.items():
        if user_id != get_user_id(image_name):
            authentication_databases[user_id] = temp_list
            temp_list = {}
        temp_list[image_name] = feature_descriptor
        user_id = get_user_id(image_name)

        if last_key == image_name:
            authentication_databases[user_id] = temp_list

    return authentication_databases

# Function to create ROC curve
def create_roc_curve(fnmr, fmr):
    # Create a new figure with the desired title
    plt.figure("ROC Curve")

    # Plot the ROC curve
    plt.title('ROC curve')
    x_axis = np.array(fmr)
    y_axis = np.array(fnmr)
    plt.plot(x_axis, y_axis)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('False Non Match Rate (FNMR)')
    plt.xlabel('False Match Rate (FMR)')
    plt.grid(True)
    plt.show()

