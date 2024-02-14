import os
import streamlit as st
import numpy as np
from scipy.ndimage import convolve
import cv2
from io import BytesIO
import base64
from main import menu


# Edge detection functions
def sobel(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply the Sobel operator for edge detection
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # Combine the gradient images to get the magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize the gradient values to the range [0, 255]
    gradient_normalized = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))

    # Thresholding to emphasize edges
    _, edge_binary = cv2.threshold(gradient_normalized, 50, 255, cv2.THRESH_BINARY)

    # Create a 3-channel image from the binary edge map
    edge_detected_image = cv2.merge([edge_binary, edge_binary, edge_binary])

    return edge_detected_image


def canny(image, min_threshold, max_threshold):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, min_threshold, max_threshold)

    # Create a 3-channel image from the binary edge map
    edge_detected_image = cv2.merge([edges, edges, edges])

    return edge_detected_image


def prewitt(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Prewitt kernels
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    # Apply Prewitt operators to compute gradients
    gradient_x = convolve(gray_image, kernel_x)
    gradient_y = convolve(gray_image, kernel_y)

    # Calculate edge strength as the magnitude of the gradient
    edge_strength = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Normalize the edge strength to [0, 255]
    edge_strength_normalized = ((edge_strength - np.min(edge_strength)) /
                                (np.max(edge_strength) - np.min(edge_strength)) * 255).astype(np.uint8)

    # Convert the edge strength back to RGB for display
    edges_rgb = cv2.cvtColor(edge_strength_normalized, cv2.COLOR_GRAY2RGB)

    return edges_rgb


# functions to save the image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def main():
    menu()
    original_image = st.session_state.original_image
    st.title("Edge Detection")
    st.write("Edge detection is a fundamental image processing technique used to identify boundaries within "
             "an image. It highlights abrupt changes in intensity, representing transitions between different "
             "structures or objects. This method is crucial for various applications, including object "
             "recognition, computer vision, and feature extraction, providing a foundation for understanding "
             "the visual content in images.")
    processed_image3 = None

    tab1, tab2, tab3 = st.tabs(["Sobel", "Canny", "Prewitt"])
    with tab1:
        st.write("Sobel edge detection is a gradient-based method widely used in image processing to identify "
                 "edges. By calculating the intensity gradients in both horizontal and vertical directions, "
                 "it emphasizes regions of rapid intensity change, effectively highlighting edges and "
                 "contours in an image.")
        sobel_button = st.button("Apply", key=1)
        if sobel_button:
            processed_image3 = sobel(original_image)
            col1, col2 = st.columns(2)
            col1.header("Original Image")
            col1.image(original_image, use_column_width=True)

            col2.header("Processed Image")
            col2.image(processed_image3, use_column_width=True)

            # Save the processed image to BytesIO using OpenCV
            processed_image_bytes = BytesIO()
            processed_image_bgr = cv2.cvtColor(processed_image3, cv2.COLOR_RGB2BGR)
            _, encoded_image = cv2.imencode(".jpg", processed_image_bgr,
                                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            processed_image_bytes.write(encoded_image)
            # Seek to the beginning of the BytesIO object
            processed_image_bytes.seek(0)
            # Download button for the processed image
            col1, col2, col3 = st.columns(3)
            with col3:
                save_image_button3 = st.download_button(
                    label="Download processed image",
                    data=processed_image_bytes,
                    mime="image/jpeg",
                    key="download_button",
                )
    with tab2:
        # Use st.form to handle the form submission
        with st.form(key='canny_form'):
            st.write("Canny edge detection is a precise algorithm for finding edges in images, using two "
                     "thresholds: a lower threshold and an upper threshold. Pixels above the upper "
                     "threshold are strong edges, those between the thresholds are weak edges, and those below "
                     "the lower threshold are non-edges. Optimal thresholding, often guided by the gradient "
                     "magnitude histogram, balances noise reduction and edge detection accuracy.")
            col_1, col_2 = st.columns(2)
            with col_1:
                min_thresh = st.slider("Set the minimum threshold", 0, 255, 100)
            with col_2:
                max_thresh = st.slider("Set the maximum threshold", 0, 255, 200)

            apply_canny_button = st.form_submit_button("Apply")

        # Process the form submission
        if apply_canny_button:
            if min_thresh >= max_thresh:
                st.error("Please set the maximum threshold to be greater than the minimum threshold")
            else:
                processed_image3 = canny(original_image, min_threshold=min_thresh, max_threshold=max_thresh)
                col1, col2 = st.columns(2)
                col1.header("Original Image")
                col1.image(original_image, use_column_width=True)

                col2.header("Processed Image")
                col2.image(processed_image3, use_column_width=True)

                # Save the processed image to BytesIO using OpenCV
                processed_image_bytes = BytesIO()
                processed_image_bgr = cv2.cvtColor(processed_image3, cv2.COLOR_RGB2BGR)
                _, encoded_image = cv2.imencode(".jpg", processed_image_bgr,
                                                [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                processed_image_bytes.write(encoded_image)
                # Seek to the beginning of the BytesIO object
                processed_image_bytes.seek(0)
                # Download button for the processed image
                col1, col2, col3 = st.columns(3)
                with col3:
                    save_image_button3 = st.download_button(
                        label="Download processed image", data=processed_image_bytes,
                        mime="image/jpeg", key="download_button",
                    )
    with tab3:
        st.write("Prewitt edge detection is a method used in images to find and emphasize edges or "
                 "boundaries. It works by looking at how the brightness of pixels changes in horizontal and "
                 "vertical directions. It uses simple mathematical operations to highlight areas where these "
                 "changes are most significant. The outcome is a map that shows where the edges in the image "
                 "are located, making it easier to identify important features and shapes")
        prewitt_button = st.button("Apply", key=2)
        if prewitt_button:
            processed_image3 = prewitt(original_image)
            col1, col2 = st.columns(2)
            col1.header("Original Image")
            col1.image(original_image, use_column_width=True)

            col2.header("Processed Image")
            col2.image(processed_image3, use_column_width=True)

            # Save the processed image to BytesIO using OpenCV
            processed_image_bytes = BytesIO()
            processed_image_bgr = cv2.cvtColor(processed_image3, cv2.COLOR_RGB2BGR)
            _, encoded_image = cv2.imencode(".jpg", processed_image_bgr,
                                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            processed_image_bytes.write(encoded_image)
            # Seek to the beginning of the BytesIO object
            processed_image_bytes.seek(0)
            # Download button for the processed image
            col1, col2, col3 = st.columns(3)
            with col3:
                save_image_button3 = st.download_button(
                    label="Download processed image",
                    data=processed_image_bytes,
                    mime="image/jpeg",
                    key="download_button",
                )



if __name__ == "__main__":
    main()