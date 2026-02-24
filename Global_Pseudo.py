# -*- coding: utf-8 -*-
"""
Created on Sat May 18 19:05:50 2024

@author: ASUS


"""

import cv2
import numpy as np
#from scipy.spatial import distance 
import math

def fill_open_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    
    for contour in contours:
        if not cv2.isContourConvex(contour):
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    result = cv2.bitwise_or(image, image, mask=closed)
    
    return result


def Euclidean_distance(vector1, vector2):
    # Ensure both vectors are of the same length
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of the same length")
    
    # Step 1: Subtract corresponding elements and square the differences
    squared_differences = [(a - b) ** 2 for a, b in zip(vector1, vector2)]
    
    # Step 2: Sum the squared differences
    sum_of_squared_differences = sum(squared_differences)
    
    # Step 3: Take the square root of the sum
    distance = math.sqrt(sum_of_squared_differences)
    
    return distance


def Calculate_feature_vector(contours, hierarchy):
    feature_vectors = []
    contour_list = []
    for i, (contour, hier) in enumerate(zip(contours, hierarchy[0])):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        bounding_rectangle_area = w * h
        
        if len(contour) < 5 or area < 5:
            continue
        
        roundness = (4 * np.pi * area) / (perimeter * perimeter)
        form_factor = area / bounding_rectangle_area
        compactness = (perimeter * perimeter) / area if area > 1 else float('inf')
        
        try:
            _, (minor_axis, major_axis), _ = cv2.fitEllipse(contour)
            axis_ratio = major_axis / minor_axis
            elongation = major_axis / bounding_rectangle_area
            moments = cv2.moments(contour)
            cx = moments['m10'] / moments['m00'] if moments['m00'] != 0 else 0
            cy = moments['m01'] / moments['m00'] if moments['m00'] != 0 else 0
            convex_hull = cv2.convexHull(contour)
            convexity_defects = cv2.convexityDefects(contour, cv2.convexHull(contour, returnPoints=False))
            orientation = np.arctan2((2 * moments['mu11']), (moments['mu20'] - moments['mu02'])) / 2
            extent = area / (w * h)
            hull_area = cv2.contourArea(convex_hull)
            solidity = area / hull_area if hull_area != 0 else 0
            (x, y), (max_w, max_h), angle = cv2.minAreaRect(contour)
            eccentricity = np.sqrt(1 - (min(max_w, max_h) / max(max_w, max_h))**2)
            feret_diameter = cv2.minEnclosingCircle(contour)[1] * np.pi
            
        except cv2.error:
            continue
        
        holes = 0
        if hier[2] != -1:
            holes = sum(1 for h in hierarchy[0] if h[3] == i)
            
            # Calculate Hu moments
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
            
            # Add each Hu moment as a separate element in the feature vector
        hu_moments_list = list(hu_moments)
        
        feature_vector = [ area, perimeter,bounding_rectangle_area,
            roundness, form_factor, compactness, axis_ratio, elongation,
            cx, cy, orientation, extent, solidity, eccentricity, feret_diameter
        ] + hu_moments_list
        feature_vectors.append(feature_vector)
        contour_list.append(contour)

    # Check the lengths of feature vectors
    #lengths = [len(vector) for vector in feature_vectors]
    #print("Lengths of feature vectors:", lengths)

    feature_vectors_array = np.array(feature_vectors)
    return feature_vectors_array, contour_list







def object_based(input_path, reference_path):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    ref_image = cv2.imread(reference_path)
    ref_image = cv2.resize(ref_image, (512, 512))
    gray = image.copy()
    gray_ref= cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    
    #edges = cv2.Canny(gray, 30, 100)
    #edges_ref = cv2.Canny(gray_ref, 30, 100)
    edges=gray
    edges_ref=gray_ref
    
    #fill contours
    # Fill open contours in the edge images
    edges = fill_open_contours(edges)
    edges_ref = fill_open_contours(edges_ref)
    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #print(len(contours))
  
    contours_ref, hierarchy_ref = cv2.findContours(edges_ref, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
    
    
    # Initialize an empty list to store feature vectors
    #feature_vectors = Calculate_feature_vector(contours, hierarchy)
    #feature_vectors_ref = Calculate_feature_vector(contours_ref, hierarchy_ref)
    output_image=ColorMap_Mean(ref_image, gray_ref, gray)
    #output_image = np.zeros_like(ref_image)
    
    #matched_contours_image = ref_image.copy()
    
    for input_contour in contours:
        area = cv2.contourArea(input_contour )
        #perimeter = cv2.arcLength(contour, True)
        #x, y, w, h = cv2.boundingRect(contour)
        #bounding_rectangle_area = w * h
        
        # Skip contour if it's too small
        if len(input_contour) < 5 or area < 5:
            #print("Contour {} skipped: Not enough points or too small".format(i+1))
            continue
        
        
        
        #cv2.drawContours(gray, [input_contour], -1, (0, 255, 0), 2)  # Draw matched contours in green
    
        best_match = None
        min_distance = float('inf')
        for ref_contour in contours_ref:
            area_ref = cv2.contourArea(ref_contour )
            #perimeter = cv2.arcLength(contour, True)
            #x, y, w, h = cv2.boundingRect(contour)
            #bounding_rectangle_area = w * h
            
            # Skip contour if it's too small
            if len(ref_contour) < 5 or area_ref < 5:
                #print("Contour {} skipped: Not enough points or too small".format(i+1))
                continue
            distance = cv2.matchShapes(input_contour, ref_contour, cv2.CONTOURS_MATCH_I1, 0.0)
            if distance < min_distance:
                min_distance = distance
                best_match = ref_contour
    
        if best_match is not None:
           #cv2.drawContours(matched_contours_image, [best_match], -1, (0, 255, 0), 2)  # Draw matched contours in green
           #print("Best match found for input contour")
            
            # Get the bounding rectangles for the input and reference contours
           x, y, w, h = cv2.boundingRect(best_match)
           x1, y1, w1, h1 = cv2.boundingRect(input_contour)
           
           
             
            
            # Iterate over each pixel in the input contour
           for i in range(x1, x1 + w1):
                for j in range(y1, y1 + h1):
                    
                    # Check if the pixel is inside the contour
                    if cv2.pointPolygonTest(input_contour, (i, j), False) > 0:
                        # Map the coordinates to the reference image and copy the color
                        ref_x = x + (i - x1)
                        ref_y = y + (j - y1)
                        if 0 <= ref_x < ref_image.shape[1] and 0 <= ref_y < ref_image.shape[0]:
                            output_image[j, i] = ref_image[ref_y, ref_x]
                        #else:
                            #print(f"Reference coordinates out of bounds: ({ref_x}, {ref_y})")
                
    #cv2.imshow("Contout_auto",gray) 
   # cv2.imshow('Matched Contours_autp_auto', matched_contours_image)      
    #cv2.imwrite('output_colored.jpg', output_image)
    #cv2.imshow('output_colored (object based_builtIn)', output_image)
    
    feature_vectors, contour_list = Calculate_feature_vector(contours, hierarchy)
    feature_vectors_ref, _ = Calculate_feature_vector(contours_ref, hierarchy_ref)
    
   # matched_contours_image = ref_image.copy()
    
    #output_image = np.zeros_like(ref_image)
    output_image=ColorMap_Mean(ref_image, gray_ref, gray)
    for input_feature_vector, input_contour in zip(feature_vectors, contour_list):
        
       # cv2.drawContours(gray, [input_contour], -1, (0, 255, 0), 2)
       # cv2.imshow("Cont",gray)
        best_match = None
        min_distance = float('inf')
        for ref_feature_vector, ref_contour in zip(feature_vectors_ref, contours_ref):
            distance = Euclidean_distance(input_feature_vector, ref_feature_vector)
            if distance < min_distance:
                min_distance = distance
                best_match = ref_contour
        
        if best_match is not None:
            #cv2.drawContours(matched_contours_image, [best_match], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(best_match)
            x1, y1, w1, h1 = cv2.boundingRect(input_contour)
            
            for i in range(x1, x1 + w1):
                for j in range(y1, y1 + h1):
                    if cv2.pointPolygonTest(input_contour, (i, j), False) > 0:
                        ref_x = x + (i - x1)
                        ref_y = y + (j - y1)
                        if 0 <= ref_x < ref_image.shape[1] and 0 <= ref_y < ref_image.shape[0]:
                            output_image[ref_y, ref_x] = ref_image[ref_y, ref_x]
    
    #cv2.imshow('Matched Contours', matched_contours_image)
    cv2.imshow('Output Colored (Object Based)', output_image)


  


def apply_global_pseudocolor(input_path, reference_path):
    # Load the input and reference images
    input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    input_img = cv2.resize(input_img, (512, 512))
    reference_img = cv2.imread(reference_path)
    reference_img = cv2.resize(reference_img, (512, 512))
    reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    
    # Generate the pseudo-colored image
    pseudo_img = ColorMap_Mean(reference_img, reference_gray, input_img)
    
    # Display the pseudo-colored image
    cv2.imshow('Pseudo-Colored Image', pseudo_img)
    
    ''' 
    # Define the start and end colors for the gradient (in BGR format)
    start_color = [0, 0, 255]  # Red
    end_color = [255, 255, 0]  # Yellow

   # Apply the color gradient for pseudocoloring
    pseudo_img = apply_color_gradient(input_img, start_color, end_color)

   # Display the original and pseudocolored images
    cv2.imshow('Original Grayscale Image', input_img)
    cv2.imshow('Pseudocolored Image', pseudo_img)
    '''
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    #return pseudo_img

def ColorMap_Mean(color_img, grey_img, inp):
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

def apply_color_gradient(input_img, start_color, end_color):
    # Create the colormap based on the gradient
    colormap = np.zeros((256, 3), dtype=np.uint8)
    color_diff = np.array(end_color) - np.array(start_color)

    for i in range(256):
        colormap[i] = np.array(start_color) + (color_diff * (i / 255))

    # Apply the colormap to the input image
    pseudo_img = np.zeros((input_img.shape[0], input_img.shape[1], 3), dtype=np.uint8)
    for x in range(input_img.shape[0]):
        for y in range(input_img.shape[1]):
            grey_shade = input_img[x, y]
            pseudo_img[x, y] = colormap[grey_shade]

    return pseudo_img



Lmax=255

def Calc_pdf_cdf(hist):
    #pdf=[]
    #cdf=[]
    #norm_cdf=[]
    pdf = hist.copy()
    cdf = hist.copy()
    norm_cdf = hist.copy()
    sum=0
    for i in range(len(hist)):
        pdf[i]=hist[i] / np.sum(hist)
        #pdf.append(hist[i] / np.sum(hist))
        sum = sum+ pdf[i]
        #cdf.append(sum)
        cdf[i]=sum
        norm = np.round(cdf[i] * Lmax)
        #norm_cdf.append(norm)
        norm_cdf[i]=norm
        
    #return pdf, cdf, norm_cdf
    return pdf, cdf



def Hist_Match(inp_cdf, erlang_cdf):
    cdf = np.ones_like(inp_cdf)
    for i in range(Lmax + 1):
        #min_dif = 1000
        #val = 0
        for j in range(Lmax + 1):
            if(inp_cdf[i]==erlang_cdf[j]):
                cdf[i] =j
                break
            elif(inp_cdf[i]<erlang_cdf[j]):
                if (abs(inp_cdf[i]-erlang_cdf[j-1])<(abs(inp_cdf[i]-erlang_cdf[j]))):
                   cdf[i] =j-1
                   break
                else:
                   cdf[i] =j
                   break
            else:
                 cdf[i] =255
                  
    return cdf
    

def Transfer(inp, new_cdf):
    output = np.ones_like(inp)
    for i in range(inp.shape[0]):
        for j in range(inp.shape[1]):
            idx = inp[i][j]
            val = new_cdf[idx]
            output[i][j] = val
    return output


def Histogram(input_path, reference_path):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    ref_image = cv2.imread(reference_path)
    ref_image = cv2.resize(ref_image, (512, 512))
    gray = image.copy()
    gray_ref= cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    #inp_size=inp.shape[0]*inp.shape[1]
    histr_inp = cv2.calcHist([image],[0],None,[Lmax+1],[0,Lmax+1])
    histr_ref = cv2.calcHist([gray_ref],[0],None,[Lmax+1],[0,Lmax+1])
    hist_inp=np.zeros(Lmax+1)
    hist_ref=np.zeros(Lmax+1)
    for i in range(Lmax+1):
        hist_inp[i]=histr_inp[i][0]
        hist_ref[i]=histr_ref[i][0]
    inp_pdf,inp_cdf=Calc_pdf_cdf(hist_inp)
    ref_pdf,ref_cdf=Calc_pdf_cdf(hist_ref)
    new_cdf=Hist_Match(inp_cdf, ref_cdf)
    output=Transfer(image, new_cdf)
    #cv2.imshow("Output",output)
    
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap_red = np.zeros(256)
    colormap_green = np.zeros(256)
    colormap_blue = np.zeros(256)
    px_count = np.zeros(256)
    
    
    colormap_jett = cv2.applyColorMap(np.arange(0, 256, dtype=np.uint8), cv2.COLORMAP_JET)


    
    # Accumulate color values and pixel counts
    for x in range(gray_ref.shape[0]):
        for y in range(gray_ref.shape[1]):
            idx = gray_ref[x, y]
            colormap_red[idx] += ref_image[x, y, 2]
            colormap_green[idx] += ref_image[x, y, 1]
            colormap_blue[idx] += ref_image[x, y, 0]
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
    pseudo_img = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
    for x in range(output.shape[0]):
        for y in range(output.shape[1]):
            grey_shade = output[x, y]
            pseudo_img[x, y] = colormap[grey_shade]

    cv2.imshow('Output Colored Histogram', pseudo_img)
    
    
    
    '''
    
    edges = cv2.Canny(output, 30, 100)
    edges_ref = cv2.Canny(gray_ref, 30, 100)
    
    #fill contours
    # Fill open contours in the edge images
    edges = fill_open_contours(edges)
    edges_ref = fill_open_contours(edges_ref)
    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #print(len(contours))
  
    contours_ref, hierarchy_ref = cv2.findContours(edges_ref, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
    
    
    # Initialize an empty list to store feature vectors
    #feature_vectors = Calculate_feature_vector(contours, hierarchy)
    #feature_vectors_ref = Calculate_feature_vector(contours_ref, hierarchy_ref)
    #output_image=ColorMap_Mean(ref_image, gray_ref, gray)
    #output_image = np.zeros_like(ref_image)
    
    #matched_contours_image = ref_image.copy()
    
    for input_contour in contours:
        area = cv2.contourArea(input_contour )
        #perimeter = cv2.arcLength(contour, True)
        #x, y, w, h = cv2.boundingRect(contour)
        #bounding_rectangle_area = w * h
        
        # Skip contour if it's too small
        if len(input_contour) < 5 or area < 5:
            #print("Contour {} skipped: Not enough points or too small".format(i+1))
            continue
        
        
        
        #cv2.drawContours(gray, [input_contour], -1, (0, 255, 0), 2)  # Draw matched contours in green
    
        best_match = None
        min_distance = float('inf')
        for ref_contour in contours_ref:
            area_ref = cv2.contourArea(ref_contour )
            #perimeter = cv2.arcLength(contour, True)
            #x, y, w, h = cv2.boundingRect(contour)
            #bounding_rectangle_area = w * h
            
            # Skip contour if it's too small
            if len(ref_contour) < 5 or area_ref < 5:
                #print("Contour {} skipped: Not enough points or too small".format(i+1))
                continue
            distance = cv2.matchShapes(input_contour, ref_contour, cv2.CONTOURS_MATCH_I1, 0.0)
            if distance < min_distance:
                min_distance = distance
                best_match = ref_contour
    
        if best_match is not None:
           #cv2.drawContours(matched_contours_image, [best_match], -1, (0, 255, 0), 2)  # Draw matched contours in green
           #print("Best match found for input contour")
            
            # Get the bounding rectangles for the input and reference contours
           x, y, w, h = cv2.boundingRect(best_match)
           x1, y1, w1, h1 = cv2.boundingRect(input_contour)
           
           
             
            
            # Iterate over each pixel in the input contour
           for i in range(x1, x1 + w1):
                for j in range(y1, y1 + h1):
                    
                    # Check if the pixel is inside the contour
                    if cv2.pointPolygonTest(input_contour, (i, j), False) > 0:
                        # Map the coordinates to the reference image and copy the color
                        ref_x = x + (i - x1)
                        ref_y = y + (j - y1)
                        if 0 <= ref_x < ref_image.shape[1] and 0 <= ref_y < ref_image.shape[0]:
                            pseudo_img[j, i] = ref_image[ref_y, ref_x]
                        #else:
                            #print(f"Reference coordinates out of bounds: ({ref_x}, {ref_y})")
                
    #cv2.imshow("Contout_auto",gray) 
   # cv2.imshow('Matched Contours_autp_auto', matched_contours_image)      
    #cv2.imwrite('output_colored.jpg', output_image)
    
    
    
   # cv2.imshow('output_colored (object-hist)', pseudo_img)
    '''


def extract_objects(gray):
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    closed_image = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def match_objects(ref_contours, input_contours,threshold=0.20):
    matches = []
    for input_cnt in input_contours:
        if cv2.contourArea(input_cnt) > 0:  # Ensure the contour is not empty
            match_scores = [cv2.matchShapes(input_cnt, ref_cnt, cv2.CONTOURS_MATCH_I1, 0.0) for ref_cnt in ref_contours]
            best_match_idx = np.argmin(match_scores)
            best_match_score = match_scores[best_match_idx]
            if best_match_score <= threshold:
               matches.append((input_cnt, ref_contours[best_match_idx]))
            #matches.append((input_cnt, ref_contours[best_match_idx]))
    return matches



def color_objects(image, matches, ref_image):
    output = np.copy(image) 
    for input_cnt, ref_cnt in matches:
        ref_x, ref_y, ref_w, ref_h = cv2.boundingRect(ref_cnt)  # Bounding rect of reference contour
        ref_obj = ref_image[ref_y:ref_y+ref_h, ref_x:ref_x+ref_w]  # Extract reference object
        input_x, input_y, input_w, input_h = cv2.boundingRect(input_cnt)  # Bounding rect of input contour
        ref_obj_resized = cv2.resize(ref_obj, (input_w, input_h))  # Resize reference object to match input object
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [input_cnt], -1, 255, thickness=cv2.FILLED)
        obj_masked = cv2.bitwise_and(output, output, mask=cv2.bitwise_not(mask))
        ref_obj_colored = cv2.bitwise_and(ref_obj_resized, ref_obj_resized, mask=mask[input_y:input_y+input_h, input_x:input_x+input_w])
        output[input_y:input_y+input_h, input_x:input_x+input_w] = cv2.add(obj_masked[input_y:input_y+input_h, input_x:input_x+input_w], ref_obj_colored)
   
    
    return output










def main(input_image_path, ref_image_col_path):
    



    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    input_image = cv2.resize(input_image, (512, 512))
    #cv2.imshow("input", input_image)

    ref_image_col = cv2.imread(ref_image_col_path)
    ref_image_col = cv2.resize(ref_image_col, (512, 512))
    
    ref_image=cv2.cvtColor(ref_image_col, cv2.COLOR_BGR2GRAY)

    ref_contours = extract_objects(ref_image)
    input_contours = extract_objects(input_image)
    
    
    # Draw contours on images for visualization
    input_contour_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    reference_contour_image = cv2.cvtColor(ref_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(input_contour_image, input_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(reference_contour_image, ref_contours, -1, (0, 255, 0), 2)

    # Display the images with detected contours
    cv2.imshow('Input Contours', input_contour_image)
    cv2.imshow('Reference Contours', reference_contour_image)

    matches = match_objects(ref_contours, input_contours)

    # Generate the pseudo-colored image from the grayscale input
    pseudo_colored_input = ColorMap_Mean(ref_image_col, ref_image, input_image)
    
    #cv2.imshow("Colored Objects_inp", pseudo_colored_input)
   

    # Color matched objects in the pseudo-colored image
    output = color_objects(pseudo_colored_input, matches, ref_image_col)

    cv2.imshow("Colored Objects", output)

   






if __name__ == "__main__":
    # Example paths (to be replaced with actual paths during the call from the main script)
    input_path = r"E:\Oishee\image\Project\Reference_Image\R2.jpg"
    reference_path = r"E:\Oishee\image\Project\Reference_Image\R2.jpg"
    apply_global_pseudocolor(input_path, reference_path)
    object_based(input_path, reference_path)
    Histogram(input_path, reference_path)
    main(input_path, reference_path)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
   
