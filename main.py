import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from io import BytesIO
import base64
import requests
import time


# Function to apply changes to the image
def apply_function(original_image, function_name):
    # Map function names to actual functions
    functions = {
        "Grayscale": grayscale,
        "Negative": negative,
        "Log": log,
        "Square": square,
        "CLAHE": clahe,
        "Enhance_contrast": enhance_contrast
    }

    # Create a copy of the original image to avoid modifying it directly
    image = np.copy(original_image)

    # Apply the selected function
    if function_name in functions:
        processed_image = functions[function_name](image)
        return processed_image
    else:
        return image


# Example image processing functions
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


# Function to apply CLAHE to an image
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


# Function to get image information
def get_image_info(image):
    info = {
        "Width": image.shape[1],
        "Height": image.shape[0],
        "Color Channels": image.shape[2] if len(image.shape) == 3 else 1,
        "Data Type": str(image.dtype),
    }
    return info


def plot_histograms(original_image):
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    color_channels = ['Red', 'Green', 'Blue']
    channel_colors = ['red', 'green', 'blue']

    for i, channel in enumerate(['r', 'g', 'b']):
        # Calculate histograms for each color channel
        hist, bins = np.histogram(original_image[:, :, i].ravel(), bins=256, range=[0, 256])
        # Calculate cumulative histograms
        cum_hist = np.cumsum(hist)

        # Plot histogram
        row = i  # Determine the row in the 3x2 grid
        col = 0  # Always use the left column
        axs[row, col].bar(bins[:-1], hist, color=channel_colors[i], alpha=0.7, width=1.0, label='Histogram')
        axs[row, col].set_title(f'{color_channels[i]} Channel Histogram')
        axs[row, col].set_xlabel('Pixel Intensity')
        axs[row, col].set_ylabel('Frequency')

        # Plot cumulative histogram
        axs[row, col + 1].plot(np.arange(256), cum_hist, color=channel_colors[i], label='Cumulative Histogram')
        axs[row, col + 1].set_title(f'{color_channels[i]} Channel Cumulative Histogram')
        axs[row, col + 1].set_xlabel('Pixel Intensity')
        axs[row, col + 1].set_ylabel('Cumulative Frequency')

    # Adjust layout
    plt.tight_layout()
    return fig


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def get_img_with_href(local_img_path, target_url, width=None):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)

    # Include width attribute if provided
    width_attr = f'width="{width}"' if width else ''

    html_code = f'''
        <a href="{target_url}" target="_blank">
            <img src="data:image/{img_format};base64,{bin_str}" {width_attr}/>
        </a>'''
    return html_code


def send_email(name, comment):
    # Mailgun API key and domain
    api_key = st.secrets["mailgun"]["api_key"]
    domain = st.secrets["mailgun"]["domain"]

    # Mailgun API endpoint
    endpoint = f'https://api.mailgun.net/v3/{domain}/messages'

    # Email parameters
    from_email = 'shahafgev10@gmail.com'
    to_email = 'shahafgev10@gmail.com'
    subject = 'New Comment from Streamlit Image Processing App'
    text = f"Name: {name}\n\nComment: {comment}"

    # Mailgun request
    response = requests.post(
        endpoint,
        auth=('api', api_key),
        data={
            'from': f'Streamlit App <{from_email}>',
            'to': [to_email],
            'subject': subject,
            'text': text
        }
    )
    # Check the response
    if response.status_code == 200:
        success_message = st.sidebar.success("Comment submitted successfully!")
        time.sleep(1)
        success_message.empty()

    else:
        st.sidebar.error("Failed to submit comment. Please try again.")


# Main Streamlit app
def main():
    # Set dark theme
    st.set_page_config(
        page_title="Image Processing App",
        page_icon=":camera:",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Set up the sidebar
    st.sidebar.title("Comment Section")
    st.sidebar.write("Let me know what you think after you play around this app.")
    # Create input box for name
    user_name = st.sidebar.text_input("Your Name:", max_chars=50)
    # Create a text area in the sidebar for the comment
    user_comment = st.sidebar.text_area("Write your comment:", max_chars=200)
    # Create a button to submit the comment
    if st.sidebar.button("Submit Comment"):
        # Check if the user has entered a name
        if not user_name:
            st.sidebar.error("Please enter your name.")
        else:
            send_email(user_name, user_comment)


    # Add GitHub and LinkedIn icons with links
    github_html = get_img_with_href('github-icon.png', 'https://github.com/shahafgev', width=30)
    linkedin_html = get_img_with_href('linkedin-icon.jpg', 'https://www.linkedin.com/in/shahaf-gev/', width=30)

    # Set the style to position the icons to the right of the title with space between them
    style = "display: flex; justify-content: flex-end; align-items: center;"

    # Add space between icons
    space_between_icons = '<div style="margin-right: 10px;"></div>'

    st.title("Image Processing App :camera:")
    st.markdown(
        f"""
            <div style="{style}">
                {github_html}
                {space_between_icons}
                {linkedin_html}
            </div>
            """,
        unsafe_allow_html=True
    )
    st.subheader("Welcome to my Image Processing App!")
    st.write("Upload your images to explore features like image information, "
             "color channel histograms, and basic transformations. Explanations will be visible once you'll upload "
             "your image.")
    st.write("More features will be added as I progress through "
             "my image processing course.")
    uploaded_file = st.file_uploader("Choose a color image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        original_image = plt.imread(uploaded_file)

        # Check the number of color channels
        if original_image.shape[-1] != 3:
            if original_image.shape[-1] == 4:
                st.warning("Note: The chosen image has an additional alpha (transparency) channel. "
                           "Consider choosing an RGB image with 3 color channels for accurate processing.")
            else:
                st.error(
                    "Error: The chosen image does not have 3 color channels. Please choose a different color image.")
            return

        with st.expander("Image information"):
            image_info = get_image_info(original_image)
            for key, value in image_info.items():
                st.write(f"{key}: {value}")

        with st.expander("Image color channels histograms"):
            st.write("Color channels histograms offer a snapshot of how different colors"
                     " contribute to an image. In an RGB image, histograms display the distribution"
                     " of pixel intensities in Red, Green, and Blue channels. Peaks in these histograms"
                     " highlight the prevalence and intensity of specific colors, aiding in"
                     " understanding the overall color composition and making adjustments"
                     " for better image quality."
                     )
            st.write("")
            # Call the function to plot histograms
            fig = plot_histograms(original_image)
            # Display the plot
            st.pyplot(fig)

        # Convert the image datatype to uint8
        original_image = original_image.astype(np.uint8)

        with st.expander("Color schemes", expanded=True):
            processed_image1 = None  # Initialize processed_image
            col1, col2 = st.columns(2)
            with col1:
                grayscale_button = st.button("Grayscale", use_container_width=True)

            with col2:
                negative_button = st.button("Negative", use_container_width=True)

            # Apply functions based on button clicks
            if grayscale_button:
                processed_image1 = apply_function(original_image, "Grayscale")
                st.write("Grayscale is a method of representing images using shades of gray. It is achieved by "
                         "removing the color information, leaving only the intensity or luminance information. In "
                         "grayscale, each pixel is represented by a single value ranging from black to white.")

            if negative_button:
                processed_image1 = apply_function(original_image, "Negative")
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
                        label="Download processed image",
                        data=processed_image_bytes,
                        mime="image/jpeg",
                        key="download_button",
                    )

        with st.expander("Image transformations", expanded=True):
            processed_image2 = None  # Initialize processed_image
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
                processed_image2 = apply_function(original_image, "Log")
                st.write("Log transformation enhances the contrast of an image by applying a logarithmic function to "
                         "its pixel values. This is particularly useful for expanding the dynamic range of "
                         "low-intensity regions, making details in darker areas more visible.")

            if square_button:
                processed_image2 = apply_function(original_image, "Square")
                st.write("The square root transformation is an image enhancement technique that involves taking the "
                         "square root of each pixel's intensity. This method is useful for improving the visibility "
                         "of details in images with low contrast, particularly in darker areas, resulting in a more "
                         "visually appealing representation.")

            if clahe_button:
                processed_image2 = apply_function(original_image, "CLAHE")
                st.write("CLAHE (Contrast Limited Adaptive Histogram Equalization) is a method for improving the "
                         "local contrast of an image. It divides the image into"
                         "small, overlapping tiles and performs histogram equalization on each tile individually. "
                         "This helps prevent over-amplification of noise in flat regions while enhancing local "
                         "details.")

            if enhance_contrast_button:
                processed_image2 = apply_function(original_image, "Enhance_contrast")
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
                        label="Download processed image",
                        data=processed_image_bytes,
                        mime="image/jpeg",
                        key="download_button",
                    )


if __name__ == "__main__":
    main()
