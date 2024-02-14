import streamlit as st
import cv2
from main import menu


# Image information function
def get_image_info(image):
    info = {
        "Width": image.shape[1],
        "Height": image.shape[0],
        "Color Channels": image.shape[2] if len(image.shape) == 3 else 1,
        "Data Type": str(image.dtype),
    }
    return info


def main():
    menu()
    original_image = st.session_state.original_image
    st.title("Image Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        image_info = get_image_info(original_image)
        for key, value in image_info.items():
            st.write(f"**{key}**: {value}")
    with col2:
        rgb_cube = cv2.imread("rgb_cube.jpg")
        st.image(rgb_cube)

    st.info("**Color channels**: In a color image, colors are typically represented using three channels: "
            "Red, Green, and Blue (RGB). Each channel contains pixel values that contribute to the overall "
            "color of the image, "
            "ranging from 0 to 255. For example, black is (0, 0, 0) and white is (255, 255, 255).",
            icon="ℹ️")


if __name__ == "__main__":
    main()