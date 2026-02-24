# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 20:41:02 2024

@author: ASUS
"""

import cv2
import numpy as np

def preprocess_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def extract_objects(image):
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def create_glcm(image, levels=256, d=1):
    max_val = levels - 1
    glcm = np.zeros((levels, levels), dtype=int)
    for i in range(image.shape[0]):
        for j in range(image.shape[1] - d):
            row = image[i, j]
            col = image[i, j + d]
            glcm[row, col] += 1
    return glcm / np.sum(glcm)

def calculate_haralick_features(glcm):
    energy = np.sum(glcm ** 2)
    contrast = 0
    homogeneity = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            contrast += (i - j) ** 2 * glcm[i, j]
            homogeneity += glcm[i, j] / (1 + abs(i - j))
    return [energy, contrast, homogeneity]

def compute_features(image, contour):
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    roi = cv2.bitwise_and(image, image, mask=mask)
    _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    glcm = create_glcm(roi_thresh)
    haralick = calculate_haralick_features(glcm)
    
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    _, _, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    hu_moments = cv2.HuMoments(cv2.moments(contour)).flatten()
    
    return [area, perimeter, aspect_ratio] + hu_moments.tolist() + haralick

def match_objects(input_features, reference_features):
    matches = []
    for i, input_feature in enumerate(input_features):
        best_match = None
        min_distance = float('inf')
        best_match_index = -1
        for j, ref_feature in enumerate(reference_features):
            distance = np.linalg.norm(np.array(input_feature) - np.array(ref_feature))
            if distance < min_distance:
                min_distance = distance
                best_match = ref_feature
                best_match_index = j
        matches.append((i, best_match_index))
    return matches

def apply_colors(input_image, input_contours, reference_contours, reference_color_image):
    colorized_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    for input_contour, reference_contour in zip(input_contours, reference_contours):
        input_x, input_y, input_w, input_h = cv2.boundingRect(input_contour)
        ref_x, ref_y, ref_w, ref_h = cv2.boundingRect(reference_contour)
        reference_color_patch = reference_color_image[ref_y:ref_y+ref_h, ref_x:ref_x+ref_w]
        resized_reference_patch = cv2.resize(reference_color_patch, (input_w, input_h), interpolation=cv2.INTER_AREA)
        input_mask = np.zeros((input_h, input_w), dtype=np.uint8)
        cv2.drawContours(input_mask, [input_contour - np.array([input_x, input_y])], -1, 255, -1)
        colorized_image[input_y:input_y+input_h, input_x:input_x+input_w] = np.where(
            input_mask[:, :, None] == 255, resized_reference_patch, colorized_image[input_y:input_y+input_h, input_x:input_x+input_w]
        )
    return colorized_image


def reduce_gray_levels(image, levels=16):
    # Scale the image to the new number of levels
    factor = 255 // levels
    reduced_image = (image // factor) * factor
    return reduced_image

# Usage example
input_image = cv2.imread(r"E:\Oishee\image\Project\Reference_Image\R1.jpg", cv2.IMREAD_GRAYSCALE)
input_image = cv2.resize(input_image, (512, 512))  # Optional resizing
input_image = reduce_gray_levels(input_image, 16)  # Reduce gray levels to speed up GLCM calculation


reference_image = cv2.imread(r"E:\Oishee\image\Project\Reference_Image\R3.jpg", cv2.IMREAD_GRAYSCALE)
reference_image = cv2.resize(reference_image, (512, 512))
reference_color_image = cv2.imread(r"E:\Oishee\image\Project\Reference_Image\R3.jpg")
reference_color_image = cv2.resize(reference_color_image, (512, 512))

input_thresh = preprocess_image(input_image)
reference_thresh = preprocess_image(reference_image)

input_contours = extract_objects(input_thresh)
reference_contours = extract_objects(reference_thresh)

input_features = [compute_features(input_image, contour) for contour in input_contours]
reference_features = [compute_features(reference_image, contour) for contour in reference_contours]

matches = match_objects(input_features, reference_features)

for input_idx, ref_idx in matches:
    print(f'Input object {input_idx} matches with reference object {ref_idx}')

matched_reference_contours = [reference_contours[ref_idx] for _, ref_idx in matches]
colorized_input_image = apply_colors(input_image, input_contours, matched_reference_contours, reference_color_image)

cv2.imshow('Colorized Image', colorized_input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
