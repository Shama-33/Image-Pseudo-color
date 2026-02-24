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
    kernel = np.ones((5,5 ), np.uint8)
    morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def compute_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    _, _, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    hu_moments = cv2.HuMoments(cv2.moments(contour)).flatten()
    return [area, perimeter, aspect_ratio] + hu_moments.tolist()

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

def apply_colors1(input_image, input_contours, reference_contours, reference_color_image):
    colorized_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    for input_contour, reference_contour in zip(input_contours, reference_contours):
        input_x, input_y, input_w, input_h = cv2.boundingRect(input_contour)
        ref_x, ref_y, ref_w, ref_h = cv2.boundingRect(reference_contour)

        # Extract the reference object's color region
        reference_color_patch = reference_color_image[ref_y:ref_y+ref_h, ref_x:ref_x+ref_w]
        # Resize the reference color patch to match the input object's size
        resized_reference_patch = cv2.resize(reference_color_patch, (input_w, input_h), interpolation=cv2.INTER_AREA)
        
        # Create a mask for the input object
        input_mask = np.zeros((input_h, input_w), dtype=np.uint8)
        cv2.drawContours(input_mask, [input_contour - np.array([input_x, input_y])], -1, 255, -1)
       
        
        # Apply the color patch to the input object area
        colorized_image[input_y:input_y+input_h, input_x:input_x+input_w] = np.where(
            input_mask[:, :, None] == 255, resized_reference_patch, colorized_image[input_y:input_y+input_h, input_x:input_x+input_w]
        )
        
    return colorized_image

def match_histograms(source, template):
    """ Adjust the pixel values of a grayscale image such that its histogram matches that of a target image """
    source_hist, bins = np.histogram(source.flatten(), 256, [0, 256])
    template_hist, bins = np.histogram(template.flatten(), 256, [0, 256])

    cdf_source = np.cumsum(source_hist) / sum(source_hist)
    cdf_template = np.cumsum(template_hist) / sum(template_hist)

    # Create a lookup table to map pixel values
    lookup_table = np.zeros(256)
    for i in range(256):
        closest_index = np.argmin(np.abs(cdf_template - cdf_source[i]))
        lookup_table[i] = closest_index

    matched = np.interp(source.flatten(), range(256), lookup_table)
    return matched.reshape(source.shape).astype(np.uint8)

def ColorMap_Mean(color_img, grey_img):
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap_red = np.zeros(256)
    colormap_green = np.zeros(256)
    colormap_blue = np.zeros(256)
    px_count = np.zeros(256)
    
    # Accumulate color values and pixel counts
    for x in range(grey_img.shape[0]):
        for y in range(grey_img.shape[1]):
            idx = grey_img[x, y]
            colormap_red[idx] += color_img[x, y, 2]
            colormap_green[idx] += color_img[x, y, 1]
            colormap_blue[idx] += color_img[x, y, 0]
            px_count[idx] += 1
    
    # Calculate mean color values
    for i in range(256):
        if px_count[i] > 0:
            colormap[i] = [
                int(colormap_blue[i] / px_count[i]), 
                int(colormap_green[i] / px_count[i]), 
                int(colormap_red[i] / px_count[i])
            ]
        else:
            colormap[i] = [100,100,200]
             
    
    return colormap

def apply_colors_using_colormap(input_grey, colormap):
    colorized = np.zeros((input_grey.shape[0], input_grey.shape[1], 3), dtype=np.uint8)
    for x in range(input_grey.shape[0]):
        for y in range(input_grey.shape[1]):
            colorized[x, y] = colormap[input_grey[x, y]]
    return colorized

def apply_colors(input_image, input_contours, reference_contours, reference_color_image):
    #colorized_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    #colorized_image_hist = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    grey_img= cv2.cvtColor(reference_color_image, cv2.COLOR_BGR2GRAY)
    colormapp=ColorMap_Mean(reference_color_image, grey_img)
    colorized_image = apply_colors_using_colormap(input_image, colormapp)
    colorized_image_hist = apply_colors_using_colormap(input_image, colormapp)

    for input_contour, reference_contour in zip(input_contours, reference_contours):
        input_x, input_y, input_w, input_h = cv2.boundingRect(input_contour)
        ref_x, ref_y, ref_w, ref_h = cv2.boundingRect(reference_contour)

        # Extract the reference object's color region and grayscale region
        reference_color_patch = reference_color_image[ref_y:ref_y+ref_h, ref_x:ref_x+ref_w]
        reference_grey_patch = cv2.cvtColor(reference_color_patch, cv2.COLOR_BGR2GRAY)
        
        

        # Create a colormap from the reference patch
        colormap = ColorMap_Mean(reference_color_patch, reference_grey_patch)

        # Extract the input object's grayscale region
        input_grey_patch = input_image[input_y:input_y+input_h, input_x:input_x+input_w]
        
        
        input_matched_patch = match_histograms(input_grey_patch, reference_grey_patch)

        # Apply the colormap to the input grayscale patch
        colorized_patch = apply_colors_using_colormap(input_grey_patch, colormap)
        
        colorized_patch_hist = apply_colors_using_colormap(input_matched_patch, colormap)
        


        # Place the colorized patch back into the colorized image
        colorized_image[input_y:input_y+input_h, input_x:input_x+input_w] = colorized_patch
        colorized_image_hist[input_y:input_y+input_h, input_x:input_x+input_w] =  colorized_patch_hist
        
    return colorized_image, colorized_image_hist




    colorized_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    grey_img = cv2.cvtColor(reference_color_image, cv2.COLOR_BGR2GRAY)
    colormap = ColorMap_Mean(reference_color_image, grey_img)

    for input_contour, reference_contour in zip(input_contours, reference_contours):
        input_x, input_y, input_w, input_h = cv2.boundingRect(input_contour)
        ref_x, ref_y, ref_w, ref_h = cv2.boundingRect(reference_contour)

        # Extract the reference and input patches
        reference_color_patch = reference_color_image[ref_y:ref_y+ref_h, ref_x:ref_x+ref_w]
        reference_grey_patch = cv2.cvtColor(reference_color_patch, cv2.COLOR_BGR2GRAY)
        input_grey_patch = cv2.cvtColor(input_image[input_y:input_y+input_h, input_x:input_x+input_w], cv2.COLOR_BGR2GRAY)

        # Resize the reference patches to match input patches if they are not the same size
        if reference_grey_patch.shape != input_grey_patch.shape:
            reference_grey_patch = cv2.resize(reference_grey_patch, (input_w, input_h), interpolation=cv2.INTER_AREA)
            reference_color_patch = cv2.resize(reference_color_patch, (input_w, input_h), interpolation=cv2.INTER_AREA)

        # Create a colorized patch based on direct matching or using colormap
        colorized_patch = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        for x in range(input_h):
            for y in range(input_w):
                if input_grey_patch[x, y] == reference_grey_patch[x, y]:
                    colorized_patch[x, y] = reference_color_patch[x, y]
                else:
                    colorized_patch[x, y] = colormap[input_grey_patch[x, y]]

        # Place the colorized patch back into the colorized image
        colorized_image[input_y:input_y+input_h, input_x:input_x+input_w] = colorized_patch
        
    return colorized_image

def main(input_image_path, ref_image_col_path):
    
    # Load images
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    input_image = cv2.resize(input_image, (512, 512))
    #reference_image = cv2.imread(r"E:\Oishee\image\Project\Reference_Image\hill1.jpg", cv2.IMREAD_GRAYSCALE)
    #reference_image = cv2.resize(reference_image, (512, 512))
    reference_color_image = cv2.imread(ref_image_col_path)
    reference_color_image = cv2.resize(reference_color_image, (512, 512))
    reference_image = cv2.cvtColor(reference_color_image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("inp",input_image)
    #cv2.imshow("ref",reference_color_image)

    # Preprocess images
    input_thresh = preprocess_image(input_image)
    reference_thresh = preprocess_image(reference_image)

    # Extract objects
    input_contours = extract_objects(input_thresh)
    reference_contours = extract_objects(reference_thresh)

    # Draw contours on images for visualization
    input_contour_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    reference_contour_image = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(input_contour_image, input_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(reference_contour_image, reference_contours, -1, (0, 255, 0), 2)

    # Display the images with detected contours
    cv2.imshow('Input Contours', input_contour_image)
    cv2.imshow('Reference Contours', reference_contour_image)

    # Compute features
    input_features = [compute_features(contour) for contour in input_contours]
    reference_features = [compute_features(contour) for contour in reference_contours]

    # Match objects
    matches = match_objects(input_features, reference_features)

    # Print matching pairs
    for input_idx, ref_idx in matches:
        print(f'Input object {input_idx} matches with reference object {ref_idx}')

    # Extract matched reference contours
    matched_reference_contours = [reference_contours[ref_idx] for _, ref_idx in matches]

    # Apply colors to input objects
    colorized_input_image, colorized_image_hist = apply_colors(input_image, input_contours, matched_reference_contours, reference_color_image)

    # Save or display the colorized image
    cv2.imshow('Colorized Image', colorized_input_image)
    cv2.imshow('Colorized Image Hist Matched',  colorized_image_hist)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
     





if __name__ == "__main__":
    # Example paths (to be replaced with actual paths during the call from the main script)
    input_path = r"E:\Oishee\image\Project\Reference_Image\R1.jpg"
    reference_path = r"E:\Oishee\image\Project\Reference_Image\R1.jpg"
    main(input_path,reference_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    