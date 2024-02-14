import os
import streamlit as st
import numpy as np
import cv2
from io import BytesIO
import base64
from main import menu


# Image transformations functions
def log(image):
    # Apply log transformation method
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(image + 1))

    # Specify the data type so that float value will be converted to int
    log_image = np.array(log_image, dtype=np.uint8)
    return log_image


def square(image):
    # Split the image into individual color channels
    b, g, r = cv2.split(image)

    # Apply square root transformation to each color channel
    sqrt_transformed_b = np.sqrt(b)
    sqrt_transformed_g = np.sqrt(g)
    sqrt_transformed_r = np.sqrt(r)

    # Normalize each transformed channel to the range [0, 255]
    sqrt_transformed_b = ((sqrt_transformed_b - np.min(sqrt_transformed_b)) /
                          (np.max(sqrt_transformed_b) - np.min(sqrt_transformed_b)) * 255).astype(np.uint8)
    sqrt_transformed_g = ((sqrt_transformed_g - np.min(sqrt_transformed_g)) /
                          (np.max(sqrt_transformed_g) - np.min(sqrt_transformed_g)) * 255).astype(np.uint8)
    sqrt_transformed_r = ((sqrt_transformed_r - np.min(sqrt_transformed_r)) /
                          (np.max(sqrt_transformed_r) - np.min(sqrt_transformed_r)) * 255).astype(np.uint8)

    # Merge the transformed channels to create the final RGB image
    sqrt_transformed_rgb = cv2.merge([sqrt_transformed_b, sqrt_transformed_g, sqrt_transformed_r])
    return sqrt_transformed_rgb


def clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    # Convert image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Extract L channel
    l_channel = lab_image[:, :, 0]

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    clahe_l = clahe.apply(l_channel)

    # Replace L channel with CLAHE-enhanced L channel
    lab_image[:, :, 0] = clahe_l

    # Convert back to RGB
    result_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

    return result_image


def enhance_contrast(image):
    # Convert the image to YCbCr color space
    ycbcr_img = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    # Compute the histogram of intensity values
    intensity_hist, bins = np.histogram(ycbcr_img[:, :, 0].ravel(), bins=256, range=[0, 256])

    # Calculate the Cumulative Distribution Function (CDF)
    cdf = np.cumsum(intensity_hist)
    cdf_normalized = cdf / cdf[-1]

    # Apply mapping to the image
    ycbcr_img[:, :, 0] = np.interp(ycbcr_img[:, :, 0], bins[:-1], cdf_normalized * 255).astype(np.uint8)

    # Revert image back to its original color scheme
    enhanced_img = cv2.cvtColor(ycbcr_img, cv2.COLOR_YCrCb2RGB)

    return enhanced_img


# functions to save the image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def main():
    menu()
    original_image = st.session_state.original_image
    st.title("Image Transformations")
    processed_image2 = None  # Initialize processed_image
    st.write("Image transformations are techniques used to enhance or modify the visual characteristics of "
             "digital images, aiming to improve their quality, visibility, or overall appearance. These "
             "transformations are applied to images for various reasons, and they play a crucial role in "
             "image processing and computer vision applications.")
    # Function buttons in the same row with a small gap
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        log_button = st.button("Log", use_container_width=True)

    with col2:
        square_button = st.button("Square", use_container_width=True)

    with col3:
        clahe_button = st.button("CLAHE", use_container_width=True)

    with col4:
        enhance_contrast_button = st.button("Enhance contrast", use_container_width=True)

    # Apply functions based on button clicks
    if log_button:
        processed_image2 = log(original_image)
        st.write("Log transformation enhances the contrast of an image by applying a logarithmic function to "
                 "its pixel values. This is particularly useful for expanding the dynamic range of "
                 "low-intensity regions, making details in darker areas more visible.")

    if square_button:
        processed_image2 = square(original_image)
        st.write("The square root transformation is an image enhancement technique that involves taking the "
                 "square root of each pixel's intensity. This method is useful for improving the visibility "
                 "of details in images with low contrast, particularly in darker areas, resulting in a more "
                 "visually appealing representation.")

    if clahe_button:
        processed_image2 = clahe(original_image)
        st.write("CLAHE (Contrast Limited Adaptive Histogram Equalization) is a method for improving the "
                 "local contrast of an image. It divides the image into"
                 "small, overlapping tiles and performs histogram equalization on each tile individually. "
                 "This helps prevent over-amplification of noise in flat regions while enhancing local "
                 "details.")

    if enhance_contrast_button:
        processed_image2 = enhance_contrast(original_image)
        st.write("Enhance Contrast using Histogram Equalization is a technique that redistributes the "
                 "intensity levels in an image, making the histogram more uniform. This enhances the overall "
                 "contrast by stretching the intensity range, making both dark and bright areas more "
                 "distinguishable. However, it may amplify noise, so methods like CLAHE are often preferred "
                 "for better results.")

    # Display both original and processed images in columns
    if processed_image2 is not None:
        col1, col2 = st.columns(2)
        col1.header("Original Image")
        col1.image(original_image, use_column_width=True)

        col2.header("Processed Image")
        col2.image(processed_image2, use_column_width=True)

        # Save the processed image to BytesIO using OpenCV
        processed_image_bytes = BytesIO()
        processed_image_bgr = cv2.cvtColor(processed_image2, cv2.COLOR_RGB2BGR)
        _, encoded_image = cv2.imencode(".jpg", processed_image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        processed_image_bytes.write(encoded_image)

        # Seek to the beginning of the BytesIO object
        processed_image_bytes.seek(0)

        # Download button for the processed image
        col1, col2, col3 = st.columns(3)
        with col3:
            save_image_button2 = st.download_button(
                label="Download processed image", data=processed_image_bytes,
                mime="image/jpeg", key="download_button", filename="my_processed_image.jpg"
            )



if __name__ == "__main__":
    main()
