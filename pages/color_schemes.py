import os
import streamlit as st
import numpy as np
import cv2
from io import BytesIO
import base64
from main import menu

# Color schemes functions
def grayscale(image):
    # Convert image to grayscale using the luminosity method
    gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    # Normalize the image to the range [0, 255]
    gray_image = (gray_image - np.min(gray_image)) / (np.max(gray_image) - np.min(gray_image)) * 255

    return gray_image.astype(np.uint8)


def negative(image):
    # Calculate the negative image using np.max
    negative_image = np.max(image) - image

    return negative_image


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def main():
    menu()
    original_image = st.session_state.original_image
    st.title("Color Schemes")
    st.write("Color schemes play a crucial role in shaping the visual language of images. Whether used for "
            "artistic expression, conveying information, or improving accessibility, color transformations "
            "provide a powerful means to tailor the color appearance of images to achieve specific "
            "objectives.")
    processed_image1 = None  # Initialize processed_image
    col1, col2 = st.columns(2)
    with col1:
        grayscale_button = st.button("Grayscale", use_container_width=True)

    with col2:
        negative_button = st.button("Negative", use_container_width=True)

    # Apply functions based on button clicks
    if grayscale_button:
        processed_image1 = grayscale(original_image)
        st.write("Grayscale is a method of representing images using shades of gray. It is achieved by "
                    "removing the color information, leaving only the intensity or luminance information. In "
                    "grayscale, each pixel is represented by a single value ranging from black to white.")

    if negative_button:
        processed_image1 = negative(original_image)
        st.write("A negative image is an inverted representation of a photograph, where the colors are "
                "reversed. In a negative, dark areas appear light, and vice versa. This transformation "
                "enhances certain details and is commonly used in photography and image processing to reveal "
                "features that may not be prominent in the original image.")

    # Display both original and processed images in columns
    if processed_image1 is not None:
        col1, col2 = st.columns(2)
        col1.header("Original Image")
        col1.image(original_image, use_column_width=True)

        col2.header("Processed Image")
        col2.image(processed_image1, use_column_width=True)

        # Save the processed image to BytesIO using OpenCV
        processed_image_bytes = BytesIO()
        processed_image_bgr = cv2.cvtColor(processed_image1, cv2.COLOR_RGB2BGR)
        _, encoded_image = cv2.imencode(".jpg", processed_image_bgr,
                                        [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        processed_image_bytes.write(encoded_image)

        # Seek to the beginning of the BytesIO object
        processed_image_bytes.seek(0)

        # Download button for the processed image
        col1, col2, col3 = st.columns(3)
        with col3:
            save_image_button1 = st.download_button(
                label="Download processed image", data=processed_image_bytes,
                mime="image/jpeg", key="download_button",
            )



if __name__ == "__main__":
    main()