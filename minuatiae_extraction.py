# import necessary libraries
import os
import cv2
import numpy as np
import operator
from matplotlib import pyplot as plt
import sys

# import user-defined functions
import fingerprint_feature_extractor
from helperFunctions import *
from fingerprint_image_enhancer import FingerprintImageEnhancer

# function to calculate similarity between minutiae
def calculate_similarity(minutiae1, minutiae2):
    # calculate orientation difference
    orientation_diff = abs(minutiae1.Orientation - minutiae2.Orientation)

    # adjust the difference to be between 0 and 180 degrees
    orientation_diff = min(orientation_diff, 180 - orientation_diff)

    # calculate similarity score inversely proportional to the orientation difference
    similarity_score = 1 - (orientation_diff / 180)

    return similarity_score


if __name__ == '__main__':
    # read enhanced images
    image_files = read_enhanced_images()

    # generate template and test data
    template_set, test_set = generate_template_and_test_data(image_files)

    print('Keys of template_set:', template_set.keys())
    print('Keys of test_set:', test_set.keys())

    # print image before enhancement
    original = cv2.imread("data/fprdata/DB2_B/101_2.tif")
    plt.imshow(original, cmap='gray')
    plt.title("Original Sample Fingerprint before enhancement")
    plt.show()

    # select a sample fingerprint
    sample_filename = "DB2_B\\101_2.tif"
    sample = test_set[sample_filename]

    # Display the original sample fingerprint
    print("Original Sample Fingerprint:")

    plt.imshow(sample, cmap='gray')
    plt.title("enhanced Sample Fingerprint")
    plt.show()

    # Detect terminations and bifurcations for the sample fingerprint
    FeaturesTerminations1, FeaturesBifurcations1 = fingerprint_feature_extractor.extract_minutiae_features(
        sample, spuriousMinutiaeThresh=10, invertImage=False, showResult=True, saveResult=True)

    # Initialize variables to store the best matching result
    best_score = 0
    best_filename = None

    # Loop through each real fingerprint image in the directory
    for counter, (fileName, file) in enumerate(template_set.items()):
        if counter % 10 == 0:
            print("Processing image", counter)

        fingerprint_img = file

        # Check if the image could not be loaded
        if fingerprint_img is None:
            print("Error loading:")
            continue

        # Extract terminations and bifurcations for the real fingerprint
        FeaturesTerminations2, FeaturesBifurcations2 = fingerprint_feature_extractor.extract_minutiae_features(
            fingerprint_img, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=True)

        # Iterate over terminations in the sample fingerprint
        for term_sample in FeaturesTerminations1:
            best_match_score = 0

            # Iterate over terminations in the real fingerprint
            for term_real in FeaturesTerminations2:
                similarity_score = calculate_similarity(term_sample, term_real)

                # Update best_match_score if the similarity_score is higher
                if similarity_score > best_match_score:
                    best_match_score = similarity_score

            # Update best_score and best_filename if best_match_score is higher
            if best_match_score > best_score:
                best_score = best_match_score
                best_filename = fileName

    # Print the closest fingerprint match and its file name
    print("Closest Fingerprint Match:")
    print("File Name:", best_filename)
    print("Similarity Score:", best_score)

    threshold = 0.8

    # Determine if the sample is accepted or rejected based on the threshold
    if best_score >= threshold:
        print("Sample Accepted")
    else:
        print("Sample Rejected")
