# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 17:00:52 2024

@author: ASUS
"""

import cv2
import numpy as np

Lmax = 255

def select_roi(image, window_name='Select ROI', colored=False):
    """ This function allows the user to select a region of interest (ROI) on an image. """
    cv2.namedWindow(window_name)
    selected_pixels = []
    temp_image = image.copy()
    if not colored:  # If the image is not colored, convert to color for visualization purposes
        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_GRAY2BGR)

    def select_roi_callback(event, x, y, flags, param):
        nonlocal selected_pixels
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_pixels.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            selected_pixels.append((x, y))

    cv2.setMouseCallback(window_name, select_roi_callback)

    while True:
        contour_image = temp_image.copy()
        if len(selected_pixels) > 0:
            cv2.polylines(contour_image, [np.array(selected_pixels)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imshow(window_name, contour_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            selected_pixels = []

    cv2.destroyAllWindows()
    return selected_pixels

def process_roi(image, selected_pixels):
    """ This function creates a mask and extracts pixels within the ROI. """
    if len(selected_pixels) > 0:
        selected_pixels_array = np.array(selected_pixels)
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.fillPoly(mask, [selected_pixels_array], 255)
        roi_pixels = image[mask == 255]
        return mask, roi_pixels
    return None, None

def calculate_histogram(roi_pixels):
    """ This function calculates the histogram of the ROI pixels. """
    histogram = np.zeros(256, dtype=int)
    for pixel in roi_pixels:
        histogram[pixel] += 1
    return histogram

def calc_cdf(hist):
    """ This function calculates the cumulative distribution function (CDF) for histogram equalization. """
    pdf = hist / np.sum(hist)
    cdf = np.cumsum(pdf)
    norm_cdf = np.round(cdf * Lmax).astype(int)
    return cdf, norm_cdf

def hist_match(inp_cdf, ref_cdf):
    """ This function creates a look-up table (LUT) based on histogram matching between two CDFs. """
    lut = np.zeros(Lmax + 1, dtype=np.uint8)
    for i in range(Lmax + 1):
        idx = np.searchsorted(ref_cdf, inp_cdf[i])
        if idx >= len(ref_cdf):
            idx = len(ref_cdf) - 1
        lut[i] = idx
    return lut



def create_colormap(ref_color_pixels, ref_gray_pixels):
    """Creates a colormap from grayscale values to RGB colors."""
    colormap = np.zeros((256, 3), dtype=np.uint8)
    color_sum = [np.zeros(3, dtype=int) for _ in range(256)]
    counts = np.zeros(256, dtype=int)

    for i in range(len(ref_gray_pixels)):
        gray_value = ref_gray_pixels[i]
        color_sum[gray_value] += ref_color_pixels[i]
        counts[gray_value] += 1

    for gray_value in range(256):
        if counts[gray_value] > 0:
            colormap[gray_value] = (color_sum[gray_value] / counts[gray_value]).astype(np.uint8)

    return colormap
def apply_colormap(output, colormap):
    """Applies a colormap to a grayscale image."""
    colorized_output = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            gray_value = output[y, x]
            colorized_output[y, x] = colormap[gray_value]
    return colorized_output

    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap_red = np.zeros(256)
    colormap_green = np.zeros(256)
    colormap_blue = np.zeros(256)
    px_count = np.zeros(256)
    
    
    colormap_jett = cv2.applyColorMap(np.arange(0, 256, dtype=np.uint8), cv2.COLORMAP_JET)


    
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
            mean_red = colormap_red[i] // px_count[i]
            mean_green = colormap_green[i] // px_count[i]
            mean_blue = colormap_blue[i] // px_count[i]
        else:
            mean_blue, mean_green, mean_red =  colormap_jett[i][0]
        
        colormap[i] = [mean_blue, mean_green, mean_red]
        
    # Create pseudo-colored image
    pseudo_img = np.zeros((inp.shape[0], inp.shape[1], 3), dtype=np.uint8)
    for x in range(inp.shape[0]):
        for y in range(inp.shape[1]):
            grey_shade = inp[x, y]
            pseudo_img[x, y] = colormap[grey_shade]

    return pseudo_img


    ref_gray_pixels, ref_color_pixels = extract_pixels(image_ref, image_ref_col, mask)

    if ref_gray_pixels is not None and ref_color_pixels is not None:
        output = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        colormap_reg = np.zeros((256, 3), dtype=np.uint8)
        colormap_red = np.zeros(256)
        colormap_green = np.zeros(256)
        colormap_blue = np.zeros(256)
        px_count = np.zeros(256)

        colormap_jett = cv2.applyColorMap(np.arange(0, 256, dtype=np.uint8), cv2.COLORMAP_JET)

        for x in range(len(ref_gray_pixels)):
            idx = ref_gray_pixels[x]
            colormap_red[idx] += ref_color_pixels[x, 2]
            colormap_green[idx] += ref_color_pixels[x, 1]
            colormap_blue[idx] += ref_color_pixels[x, 0]
            px_count[idx] += 1

        for i in range(256):
            if px_count[i] > 0:
                mean_red = colormap_red[i] // px_count[i]
                mean_green = colormap_green[i] // px_count[i]
                mean_blue = colormap_blue[i] // px_count[i]
            else:
                mean_blue, mean_green, mean_red = colormap_jett[i][0]
            colormap_reg[i] = [mean_blue, mean_green, mean_red]

        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x] == 255:
                    grey_shade = image[y, x]
                    output[y, x] = colormap_reg[grey_shade]

        return output, colormap_reg

    return None, None


def main(input_image_path, ref_image_col_path):
    """ Main function to execute the histogram matching process. """
    # Load images
    image_input = cv2.imread(input_image_path, 0)  # Load in grayscale
    image_input = cv2.resize(image_input, (512, 512))
    image_ref_col = cv2.imread(ref_image_col_path, 1)  # Load in color
    image_ref_col = cv2.resize(image_ref_col, (512, 512))
    image_ref = cv2.cvtColor(image_ref_col, cv2.COLOR_BGR2GRAY)

    # ROI selection on the input image
    print("Select region of interest in the input image.")
    selected_pixels_input = select_roi(image_input, 'Input Image', colored=False)
    mask_input, roi_pixels_input = process_roi(image_input, selected_pixels_input)

    if roi_pixels_input is not None:
        histogram_input = calculate_histogram(roi_pixels_input)

        # ROI selection on the reference image
        print("Select corresponding region of interest in the reference image.")
        selected_pixels_ref = select_roi(image_ref_col, 'Reference Image', colored=True)
        mask_ref, roi_pixels_ref = process_roi(image_ref, selected_pixels_ref)

        if roi_pixels_ref is not None:
            histogram_ref = calculate_histogram(roi_pixels_ref)

            inp_cdf, inp_norm_cdf = calc_cdf(histogram_input)
            ref_cdf, ref_norm_cdf = calc_cdf(histogram_ref)
            
            new_cdf = hist_match(inp_norm_cdf, ref_norm_cdf)

            output = np.zeros_like(image_input)
            for y in range(mask_input.shape[0]):
                for x in range(mask_input.shape[1]):
                    if mask_input[y, x] == 255:
                        output[y, x] = new_cdf[image_input[y, x]]

            #cv2.imshow("Output_region_mean", output)
            # Create colormap and apply to the histogram-matched output
            ref_color_pixels = image_ref_col[mask_ref == 255]
            ref_gray_pixels = image_ref[mask_ref == 255]
            colormap = create_colormap(ref_color_pixels, ref_gray_pixels)
            colorized_output = apply_colormap(output, colormap)

           # Display the results
            cv2.imshow("Histogram Matched and Colorized Output", colorized_output)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No region of interest selected in the reference image.")
    else:
        print("No region of interest selected in the input image.")
        
        
 

if __name__ == "__main__":
    input_image_path = r"E:\Oishee\image\Project\Reference_Image\hill1.jpg"
    ref_image_col_path = r"E:\Oishee\image\Project\Reference_Image\hill2.jpg"
    main(input_image_path, ref_image_col_path)
