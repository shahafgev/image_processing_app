import os
import streamlit as st
import numpy as np
import cv2
from io import BytesIO
import base64
from main import menu


# Image transformations functions
def erosion(image, kernel_size, lower_threshold, upper_threshold, iterations=1):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(gray_image, lower_threshold, upper_threshold, cv2.THRESH_BINARY_INV)
    image_binary = np.uint8(threshold_image)

    kernel = np.ones((kernel_size, kernel_size))
    image_erosion = cv2.erode(image_binary, kernel, iterations)
    return image_erosion


def dilation(image, kernel_size, lower_threshold, upper_threshold, iterations=1):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(gray_image, lower_threshold, upper_threshold, cv2.THRESH_BINARY_INV)
    image_binary = np.uint8(threshold_image)

    kernel = np.ones((kernel_size, kernel_size))
    image_dilation = cv2.dilate(image_binary, kernel, iterations)
    return image_dilation


def opening(image, kernel_size, lower_threshold, upper_threshold):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(gray_image, lower_threshold, upper_threshold, cv2.THRESH_BINARY_INV)
    image_binary = np.uint8(threshold_image)

    kernel = np.ones((kernel_size, kernel_size))
    # Opening (Erosion followed by Dilation)
    image_opening = cv2.morphologyEx(image_binary, cv2.MORPH_OPEN, kernel)
    return image_opening


def closing(image, kernel_size, lower_threshold, upper_threshold):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(gray_image, lower_threshold, upper_threshold, cv2.THRESH_BINARY_INV)
    image_binary = np.uint8(threshold_image)

    kernel = np.ones((kernel_size, kernel_size))
    # Closing (Dilation followed by Erosion)
    image_closing = cv2.morphologyEx(image_binary, cv2.MORPH_CLOSE, kernel)
    return image_closing


def morphological_gradient(image, kernel_size, lower_threshold, upper_threshold, iterations=1):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(gray_image, lower_threshold, upper_threshold, cv2.THRESH_BINARY_INV)
    image_binary = np.uint8(threshold_image)

    kernel = np.ones((kernel_size, kernel_size))
    # Morphological Gradient (Dilation - Erosion)
    image_erosion = cv2.erode(image_binary, kernel, iterations=iterations)
    image_dilation = cv2.dilate(image_binary, kernel, iterations=iterations)
    morphological_grad = cv2.subtract(image_dilation, image_erosion)
    return morphological_grad


# functions to save the image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def main():
    menu()
    original_image = st.session_state.original_image

    st.title("Morphological operations")
    processed_image = None  # Initialize processed_image
    st.write("Morphological operations in image processing are fundamental techniques that manipulate the shape and "
             "structure of objects within an image. These operations, such as dilation, erosion, and opening/closing, "
             "are employed to enhance or suppress certain features, aiding in tasks like noise reduction, "
             "object segmentation, and boundary extraction. By altering pixel values based on the local "
             "characteristics of an image, morphological operations play a crucial role in improving the overall "
             "quality and interpretability of processed images.")
    st.write("Morphological operations are commonly used in images of text for tasks like noise reduction and segmentation."
             " They are also applied in medical imaging for blood vessel extraction and tumor detection, in industrial"
             " settings for defect detection and object counting, and in satellite imagery for tasks like land cover"
             " classification and road extraction.")

    # Function buttons in the same row with a small gap
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Erosion", "Dilation", "Opening", "Closing", "Morphological gradient"])

    with tab1:
        # Use st.form to handle the form submission
        with st.form(key='erosion_form'):
            st.write("Erosion in image processing is a morphological operation that shrinks the boundaries of objects "
                     "within an image. It works by moving a structuring element (a kernel) across the image and "
                     "replacing each pixel with the minimum pixel value within the kernel's neighborhood. Erosion is "
                     "often used for noise reduction and separating or shrinking connected objects.")
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                min_thresh = st.slider("Set the minimum threshold", 0, 255, 100, key="erosion1")
            with col_2:
                max_thresh = st.slider("Set the maximum threshold", 0, 255, 200, key="erosion2")
            with col_3:
                kernel_size = st.slider("Set the kerenel size", 3, 15, 3, step=2, key="erosion3")

            apply_erosion_button = st.form_submit_button("Apply")

        # Process the form submission
        if apply_erosion_button:
            if min_thresh >= max_thresh:
                st.error("Please set the maximum threshold to be greater than the minimum threshold")
            else:
                processed_image = erosion(original_image, kernel_size=kernel_size,
                                          lower_threshold=min_thresh, upper_threshold=max_thresh)

                col1, col2 = st.columns(2)
                col1.header("Original Image")
                col1.image(original_image, use_column_width=True)

                col2.header("Processed Image")
                col2.image(processed_image, use_column_width=True)

                # Save the processed image to BytesIO using OpenCV
                processed_image_bytes = BytesIO()
                processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
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
    with tab2:
        # Use st.form to handle the form submission
        with st.form(key='dilation_form'):
            st.write("Dilation is a morphological operation that expands the boundaries of objects in an image. It "
                     "involves moving a structuring element across the image and replacing each pixel with the "
                     "maximum pixel value within the kernel's neighborhood. Dilation is useful for joining broken "
                     "structures, enlarging objects, and enhancing features in an image.")
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                min_thresh = st.slider("Set the minimum threshold", 0, 255, 100, key="dilation1")
            with col_2:
                max_thresh = st.slider("Set the maximum threshold", 0, 255, 200, key="dilation2")
            with col_3:
                kernel_size = st.slider("Set the kerenel size", 3, 15, 3, step=2, key="dilation3")

            apply_dilation_button = st.form_submit_button("Apply")

        # Process the form submission
        if apply_dilation_button:
            if min_thresh >= max_thresh:
                st.error("Please set the maximum threshold to be greater than the minimum threshold")
            else:
                processed_image = dilation(original_image, kernel_size=kernel_size,
                                           lower_threshold=min_thresh, upper_threshold=max_thresh)
                col1, col2 = st.columns(2)
                col1.header("Original Image")
                col1.image(original_image, use_column_width=True)

                col2.header("Processed Image")
                col2.image(processed_image, use_column_width=True)

                # Save the processed image to BytesIO using OpenCV
                processed_image_bytes = BytesIO()
                processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
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
        # Use st.form to handle the form submission
        with st.form(key='opening_form'):
            st.write("Opening is a combination of erosion followed by dilation. It is used to remove small objects "
                     "and smooth the contours of larger objects. Opening is particularly effective in eliminating "
                     "noise and fine details while preserving the essential structure of larger elements in an image.")
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                min_thresh = st.slider("Set the minimum threshold", 0, 255, 100, key="opening1")
            with col_2:
                max_thresh = st.slider("Set the maximum threshold", 0, 255, 200, key="opening2")
            with col_3:
                kernel_size = st.slider("Set the kerenel size", 3, 15, 3, step=2, key="opening3")

            apply_opening_button = st.form_submit_button("Apply")

        # Process the form submission
        if apply_opening_button:
            if min_thresh >= max_thresh:
                st.error("Please set the maximum threshold to be greater than the minimum threshold")
            else:
                processed_image = opening(original_image, kernel_size=kernel_size,
                                          lower_threshold=min_thresh, upper_threshold=max_thresh)
                col1, col2 = st.columns(2)
                col1.header("Original Image")
                col1.image(original_image, use_column_width=True)

                col2.header("Processed Image")
                col2.image(processed_image, use_column_width=True)

                # Save the processed image to BytesIO using OpenCV
                processed_image_bytes = BytesIO()
                processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
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
    with tab4:
        # Use st.form to handle the form submission
        with st.form(key='closing_form'):
            st.write("Closing is the opposite of opening and consists of dilation followed by erosion. This operation "
                     "is useful for closing small gaps and connecting small breaks in contours. Closing is effective "
                     "in filling holes and completing the shapes of objects, making it valuable in tasks such as "
                     "object segmentation and boundary refinement.")
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                min_thresh = st.slider("Set the minimum threshold", 0, 255, 100, key="closing1")
            with col_2:
                max_thresh = st.slider("Set the maximum threshold", 0, 255, 200, key="closing2")
            with col_3:
                kernel_size = st.slider("Set the kerenel size", 3, 15, 3, step=2, key="closing3")
            apply_closing_button = st.form_submit_button("Apply")

        # Process the form submission
        if apply_closing_button:
            if min_thresh >= max_thresh:
                st.error("Please set the maximum threshold to be greater than the minimum threshold")
            else:
                processed_image = closing(original_image, kernel_size=kernel_size,
                                          lower_threshold=min_thresh, upper_threshold=max_thresh)
                col1, col2 = st.columns(2)
                col1.header("Original Image")
                col1.image(original_image, use_column_width=True)

                col2.header("Processed Image")
                col2.image(processed_image, use_column_width=True)

                # Save the processed image to BytesIO using OpenCV
                processed_image_bytes = BytesIO()
                processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
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
    with tab5:
        # Use st.form to handle the form submission
        with st.form(key='morphological_gradient_form'):
            st.write("The morphological gradient is an operation that highlights the boundaries of objects in an "
                     "image. It is obtained by taking the difference between the dilation and erosion of an image. "
                     "The result emphasizes regions of rapid intensity change, making it useful for edge detection "
                     "and extracting key features such as object boundaries or contours in image analysis.")
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                min_thresh = st.slider("Set the minimum threshold", 0, 255, 100, key="morph1")
            with col_2:
                max_thresh = st.slider("Set the maximum threshold", 0, 255, 200, key="morph2")
            with col_3:
                kernel_size = st.slider("Set the kerenel size", 3, 15, 3, step=2, key="morph3")


            apply_morphological_gradient_button = st.form_submit_button("Apply")

        # Process the form submission
        if apply_morphological_gradient_button:
            if min_thresh >= max_thresh:
                st.error("Please set the maximum threshold to be greater than the minimum threshold")
            else:
                processed_image = morphological_gradient(original_image, kernel_size=kernel_size,
                                                         lower_threshold=min_thresh, upper_threshold=max_thresh)
                col1, col2 = st.columns(2)
                col1.header("Original Image")
                col1.image(original_image, use_column_width=True)

                col2.header("Processed Image")
                col2.image(processed_image, use_column_width=True)

                # Save the processed image to BytesIO using OpenCV
                processed_image_bytes = BytesIO()
                processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
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


if __name__ == "__main__":
    main()
