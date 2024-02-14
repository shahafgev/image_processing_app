import os
import streamlit as st
import matplotlib.pyplot as plt
import base64
import requests
import time


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


# Comment section
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


def menu():
    st.sidebar.page_link("main.py", label="Main Page")
    # Check if original_image is available in session state
    if 'original_image' in st.session_state and st.session_state.original_image is not None:
        st.sidebar.page_link("pages/image_information.py", label="Image Information")
        st.sidebar.page_link("pages/color_channels_histograms.py", label="Color Channels Histograms")
        st.sidebar.page_link("pages/color_schemes.py", label="Color Schemes")
        st.sidebar.page_link("pages/image_transformations.py", label="Image Transformations")
        st.sidebar.page_link("pages/edge_detection.py", label="Edge Detection")


# Set dark theme
st.set_page_config(
    page_title="Image Processing App",
    page_icon=":camera:",
    layout="centered",
    initial_sidebar_state="expanded"
)


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

# Initialize session state
if 'original_image' not in st.session_state:
    st.session_state.original_image = None


# Main Streamlit app
def main():
    st.subheader("Welcome to my Image Processing App!")
    st.write("Upload your images to explore features like image transformations, edge detection and etc.")
    st.write("The side menu will expand to reveal the various options after you upload your photo. By clicking again "
             "on the main page in the menu, your photo will be deleted and you can upload another photo if you wish.")
    st.write("More features will be added as I progress through "
             "my image processing course.")

    uploaded_file = st.file_uploader("Choose a color image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Store the original_image in session state
        st.session_state.original_image = plt.imread(uploaded_file)

        # Check the number of color channels
        if st.session_state.original_image.shape[-1] != 3:
            if st.session_state.original_image.shape[-1] == 4:
                st.warning("Note: The chosen image has an additional alpha (transparency) channel. "
                           "Consider choosing an RGB image with 3 color channels for accurate processing.")
            else:
                st.error(
                    "Error: The chosen image does not have 3 color channels. Please choose a different color image.")
            return

        menu()

    # Set up the sidebar
    with st.expander("Comment Section"):
        st.subheader("Did you like this app?")
        st.write("Let me know what you think after you play around twith your image. \n"
                 "After you'll press the submit comment button I will receive an email with your comment")
        # Create input box for name
        user_name = st.text_input("Your Name:", max_chars=50)
        # Create a text area in the sidebar for the comment
        user_comment = st.text_area("Write your comment:", max_chars=200)
        # Create a button to submit the comment
        if st.button("Submit Comment"):
            # Check if the user has entered a name
            if not user_name:
                st.error("Please enter your name.")
            else:
                send_email(user_name, user_comment)


if __name__ == "__main__":
    main()
