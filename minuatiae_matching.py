import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from sklearn.metrics import confusion_matrix, accuracy_score
from helperFunctions import read_enhanced_images, generate_template_and_test_data, authentication_dataset_split, get_user_id, get_genuine_impostor_scores, create_roc_curve
from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report
from tabulate import tabulate


# Initialize SIFT detector for matching keypoints
sift = cv2.SIFT_create(500)

# Initialize Brute-Force Matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Function to find best matches between test image and template features
def find_best_matches(test_image, template_features, distance_threshold):
    best_matches_dict = {}
    kp1, test_des = sift.detectAndCompute(test_image, None)

    for template_image_id, template_feature_des in template_features.items():
        if test_des is not None and template_feature_des is not None:
            matches = bf.match(test_des, template_feature_des)
            matches = list(matches)
            matches.sort(key=lambda x: x.distance, reverse=False)
            best_matches = sum(1 for m in matches if m.distance < distance_threshold)
            best_matches_dict[template_image_id] = best_matches

    best_matches_dict = sorted(best_matches_dict.items(), key=lambda x: x[1], reverse=True)
    return best_matches_dict

# Function to draw matches between two fingerprint images
def draw_matches(image1, image2):
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    matches = list(bf.match(descriptors1, descriptors2))
    matches.sort(key=lambda x: x.distance, reverse=False)
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:70], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Matches", matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to count close fingerprints
def num_close_fingerprints(feature_distances, len_best_matches):
    count_same = sum(1 for features in feature_distances if int(features[1]) > len_best_matches)
    return count_same

# Function for authentication
def authenticate(dist_threshold, len_best_matches):
    true_users = []
    predicted_users = []
    total_distances = []

    for auth_key, auth_database in authentication_dataset.items():
        for test_image_name, test_image in test_set.items():
            best_matches_dict = find_best_matches(test_image, auth_database, dist_threshold)
            total_distances.append(best_matches_dict[0])

            count_same = num_close_fingerprints(best_matches_dict, len_best_matches)
            matching_ratio = count_same / len(auth_database)

            true_user = int(get_user_id(test_image_name) == auth_key)
            true_users.append(true_user)

            predicted_user = int(matching_ratio >= 0.3)
            predicted_users.append(predicted_user)
    print("At distance: " + str(dist_threshold));
    print("Accuracy is %f " % (round(accuracy_score(true_users, predicted_users), 4)))

    return total_distances, true_users, predicted_users

def show_confusion_matrix(conf_matrix):
    true_negative = conf_matrix[0, 0]
    false_positive = conf_matrix[0, 1]
    false_negative = conf_matrix[1, 0]
    true_positive = conf_matrix[1, 1]

    print("Confusion Matrix:")
    print("True Negative:", true_negative)
    print("False Positive:", false_positive)
    print("False Negative:", false_negative)
    print("True Positive:", true_positive)


def calculate_fmr_fnmr(distances, true_users, threshold):
    # Initialize counters for false matches and false non-matches
    false_matches = 0
    false_non_matches = 0

    # Loop through the distances and true user labels
    for distance_tuple, true_user in zip(distances, true_users):
        distance = distance_tuple[1]  # Extract the distance value from the tuple
        # If distance is less than threshold and true user is 0 (impostor), it's a false match
        if distance < threshold and true_user == 0:
            false_matches += 1
        # If distance is greater than or equal to threshold and true user is 1 (genuine), it's a false non-match
        elif distance >= threshold and true_user == 1:
            false_non_matches += 1

    # Calculate False Match Rate (FMR) and False Non-Match Rate (FNMR)
    fmr = false_matches / len(true_users)
    fnmr = false_non_matches / len(true_users)

    return fmr, fnmr


# Read enhanced images
image_files = read_enhanced_images()

# Generate template and test data
template_set, test_set = generate_template_and_test_data(image_files)

# Split authentication dataset
authentication_dataset = authentication_dataset_split(template_set)

# Print useres in the template set
print('Users in the template set = {}'.format(authentication_dataset.keys()))

# Example usage
test_image_name = list(test_set.keys())[0]
test_image = test_set[test_image_name]
authentication_db = authentication_dataset['B\\102']

best_matches = find_best_matches(test_image, authentication_db, 40)
num_close = num_close_fingerprints(best_matches, 10)
probability = num_close / len(authentication_db)
# print(f'For test image for the entered user: {test_image_name}')
# print('Probability of correct fingerprint for user 102 = {:.4f}'.format(probability))

template_image = template_set['DB2_B\\101_1.tif']
draw_matches(test_image, template_image)

# Authentication
num_best_matches = 15
distance_thresholds = range(30, 70, 10)

for threshold in distance_thresholds:
    authenticate(threshold, num_best_matches)

distances, true_users, predicted_users = authenticate(40, num_best_matches)
genuine_scores, impostor_scores = get_genuine_impostor_scores(distances, true_users)
eer_stats = get_eer_stats(genuine_scores, impostor_scores)
generate_eer_report([eer_stats], ['A'], 'pyeer_report.csv')

equal_error_rate = round(eer_stats.eer, 4)
print('Equal Error Rate (EER): {:.4f}'.format(equal_error_rate))

threshold = 0.3
fmr, fnmr = calculate_fmr_fnmr(distances, true_users, threshold)
print("False Match Rate (FMR):", fmr)
print("False Non-Match Rate (FNMR):", fnmr)

# Print the confusion matrix and accuracy
conf_matrix = confusion_matrix(true_users, predicted_users)
show_confusion_matrix(conf_matrix)

accuracy = accuracy_score(true_users, predicted_users)
print('Accuracy:', accuracy)


# Plot the ROC curve
print('Generating ROC curve: False match rates vs. false non-match rates')
create_roc_curve(eer_stats.fnmr, eer_stats.fmr)


