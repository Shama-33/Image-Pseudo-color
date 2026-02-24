import cv2
import numpy as np

def select_roi(image):
    cv2.namedWindow('Select ROI')
    selected_pixels = []

    def select_roi_callback(event, x, y, flags, param):
        nonlocal selected_pixels
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_pixels.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            selected_pixels.append((x, y))

    cv2.setMouseCallback('Select ROI', select_roi_callback)

    while True:
        contour_image = np.copy(image)
        if len(selected_pixels) > 0:
            cv2.polylines(contour_image, [np.array(selected_pixels)], isClosed=True, color=(0, 0, 255), thickness=1)

        cv2.imshow('Select ROI', contour_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            selected_pixels = []

    cv2.destroyAllWindows()
    return selected_pixels

def process_roi(image, selected_pixels):
    if len(selected_pixels) > 0:
        selected_pixels_array = np.array(selected_pixels)
        im = image.copy()
        roi_contour = np.array([selected_pixels_array], dtype=np.int32)
        mask = np.zeros_like(im)
        cv2.fillPoly(mask, [roi_contour], 255)
        roi_pixels = image[mask == 255]
        return mask, roi_pixels
    return None, None

def extract_pixels(image_ref, image_ref_col, mask):
    if mask is not None:
        ref_gray_pixels = image_ref[mask == 255]
        ref_color_pixels = image_ref_col[mask == 255]
        return ref_gray_pixels, ref_color_pixels
    return None, None

def colormap_mean(image, image_ref_col, image_ref, mask):
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

Lmax = 255

def calculate_histogram(roi_pixels):
    histogram = np.zeros(256, dtype=int)
    for pixel in roi_pixels:
        histogram[pixel] += 1
    return histogram

def calc_cdf(hist):
    pdf = hist / np.sum(hist)
    cdf = np.cumsum(pdf)
    norm_cdf = np.round(cdf * Lmax).astype(int)
    return cdf, norm_cdf

def hist_match(inp_cdf, erlang_cdf):
    lut = np.zeros(Lmax + 1, dtype=np.uint8)
    for i in range(Lmax + 1):
        idx = np.searchsorted(erlang_cdf, inp_cdf[i])
        if idx >= len(erlang_cdf):
            idx = len(erlang_cdf) - 1
        lut[i] = idx
    return lut

def main(input_image_path, ref_image_col_path):
    image = cv2.imread(input_image_path, 0)
    image = cv2.resize(image, (512, 512))

    image_ref_col = cv2.imread(ref_image_col_path, 1)
    image_ref_col = cv2.resize(image_ref_col, (512, 512))

    image_ref = cv2.cvtColor(image_ref_col, cv2.COLOR_BGR2GRAY)

    selected_pixels = select_roi(image)
    mask, roi_pixels = process_roi(image, selected_pixels)

    if roi_pixels is not None:
        histogram_input = calculate_histogram(roi_pixels)
        ref_gray_pixels, _ = extract_pixels(image_ref, image_ref_col, mask)
        histogram_ref = calculate_histogram(ref_gray_pixels)

        inp_cdf, inp_norm_cdf = calc_cdf(histogram_input)
        ref_cdf, ref_norm_cdf = calc_cdf(histogram_ref)
        
        new_cdf = hist_match(inp_cdf, ref_cdf)

        output = np.zeros_like(image)
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x] == 255:
                    output[y, x] = new_cdf[image[y, x]]

        output_norm, colormap_reg = colormap_mean(image, image_ref_col, image_ref, mask)
        
        output2 = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x] == 255:
                    grey_shade = output[y, x]
                    output2[y, x] = colormap_reg[grey_shade]

        cv2.imshow("Output_region_mean", output_norm)
        cv2.imshow("Match Hist", output2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No region of interest selected.")

if __name__ == "__main__":
    input_image_path = r"E:\Oishee\image\Project\Reference_Image\hill1.jpg"
    ref_image_col_path = r"E:\Oishee\image\Project\Reference_Image\hill2.jpg"
    main(input_image_path, ref_image_col_path)
