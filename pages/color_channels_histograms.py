import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from main import menu


# Plot image histograms function
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


def main():
    menu()
    original_image = st.session_state.original_image
    st.title("Color Channels Histograms")
    st.write("Color channels histograms offer a snapshot of how different colors"
             " contribute to an image. In an RGB image, histograms display the distribution"
             " of pixel intensities in Red, Green, and Blue channels. Peaks in these histograms"
             " highlight the prevalence and intensity of specific colors, aiding in"
             " understanding the overall color composition and making adjustments"
             " for better image quality. \n"
             )

    # Call the function to plot histograms
    fig = plot_histograms(original_image)
    # Display the plot
    st.pyplot(fig)
    st.write("")

    st.write("Click the tabs to see what we can learn:")
    tab1, tab2 = st.tabs(["Histograms", "Cumulative histograms"])
    with tab1:
        st.write(" 1. **Brightness/Exposure:**\n"
                 "A peak shifted to the right side of the histogram indicates brighter or "
                 "overexposed areas,"
                 "while a peak on the left suggests darker or underexposed regions.\n"
                 " 2. **Contrast:**\n"
                 "A wider spread of values across the histogram suggests a higher contrast in the "
                 "image,"
                 "while a narrower range may indicate lower contrast.\n"
                 " 3. **Saturation:**\n"
                 "In a color image, a color channel with a broader distribution indicates a more "
                 "saturated"
                 "presence of that color in the image. A narrow distribution may suggest "
                 "desaturation.\n"
                 " 4. **Image Quality:**\n"
                 "A well-distributed histogram with values spread across the entire range often "
                 "indicates a"
                 "high-quality image with a broad range of tones and colors.\n"
                 " 5. **Artifacts or Anomalies:**\n"
                 "Unusual patterns or spikes in a color channel may indicate artifacts, noise, "
                 "or specific"
                 "features in the image that could require attention.")
    with tab2:
        st.write(" 1. **Overall Brightness Distribution:**\n"
                 "The cumulative histogram allows you to see the cumulative distribution of "
                 "brightness values in the image. Steeper inclines indicate regions with higher "
                 "pixel intensities, while flatter sections represent lower intensities.\n"
                 "2. **Clipping and Saturation:**\n"
                 "Flat sections at the extremes of the cumulative histogram may suggest areas where "
                 "pixel values are saturated or clipped. This can help in identifying overexposed or "
                 "underexposed regions.\n"
                 " 3. **Saturation Levels:**\n"
                 "In a color image, examining the cumulative histograms of individual color channels "
                 "can help assess the overall saturation levels. Steeper slopes may suggest more "
                 "vivid colors, while flatter slopes indicate desaturation.")



if __name__ == "__main__":
    main()

