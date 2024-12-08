import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd


def detect_crops_and_health_week2(image, min_pixels=900):
    """
    Detect crops, calculate green pixel count, and assess health for Week 2.
    Returns processed image, crop health status, and improved mask.
    """
    # Convert to HSV for green segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Morphological closing to merge close regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    closed_green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # Find connected components in the refined mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_green_mask, connectivity=8)

    # Minimum size to consider a valid crop
    min_component_area = 500

    # Initialize results
    processed_image = image.copy()
    crop_health = []
    total_crops = 0
    healthy_crops = 0

    for i in range(1, num_labels):  # Start from 1 to skip the background
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_component_area:
            total_crops += 1
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[
                i, cv2.CC_STAT_HEIGHT]

            # Calculate green pixel count in the bounding box
            crop_mask = closed_green_mask[y:y + h, x:x + w]
            green_pixels = cv2.countNonZero(crop_mask)
            total_pixels = w * h
            green_percentage = (green_pixels / total_pixels) * 100

            # Determine health status for Week 2 (40% threshold and min_pixels)
            if green_percentage >= 40 and green_pixels >= min_pixels:
                health_status = "Healthy"
                healthy_crops += 1
                border_color = (0, 255, 0)  # Green border for Healthy crops
            else:
                health_status = "Unhealthy"
                border_color = (255, 0, 0)  # Red border for Unhealthy crops

            # Draw the border and label crop number
            cv2.rectangle(processed_image, (x, y), (x + w, y + h), border_color, 3)
            cv2.putText(processed_image, f"Crop {total_crops}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        border_color, 2)

            # Append results
            crop_health.append({
                "Crop": total_crops,
                "Health Status": health_status
            })

    # Create an improved mask with white background and green crops
    improved_mask = np.ones_like(image) * 255  # Initialize mask with white background
    improved_mask[closed_green_mask > 0] = [0, 255, 0]  # Set green areas to green

    return processed_image, crop_health, healthy_crops, total_crops, improved_mask


# Streamlit app
st.set_page_config(page_title="Crop Health Dashboard - Week 2", layout="wide")
st.title("Crop Health Detection Dashboard - Week 2")
st.write("Analyze crop health based on canopy coverage and minimum green pixels for Week 2.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], key="unique_key_2")

if uploaded_file is not None:
    # Load the uploaded image
    image = np.array(Image.open(uploaded_file))

    # Detect crops and calculate health for Week 2
    processed_image, crop_health, healthy_crops, total_crops, improved_mask = detect_crops_and_health_week2(image)

    # Create a three-column layout
    img_col1, img_col2, report_col = st.columns([1, 1, 1.5])  # Adjust column widths

    # Display Original and Processed Images side by side
    with img_col1:
        st.image(image, caption="Original Image", width=300)

    with img_col2:
        st.image(processed_image, caption="Processed Image", width=300)

    # Display Overall Summary and Dropdown in the Right Column
    with report_col:
        st.subheader("Overall Summary")

        # Display overall statistics as a table
        summary_data = {
            "Metric": ["Total Crops Detected", "Healthy Crops", "Unhealthy Crops"],
            "Value": [total_crops, healthy_crops, total_crops - healthy_crops]
        }
        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)

        # Dropdown to select a specific crop for details
        st.subheader("Crop Details")
        crop_options = [f"Crop {crop['Crop']}" for crop in crop_health]
        selected_crop = st.selectbox("Select a Crop for Detailed Information:", crop_options)

        # Display selected crop details
        if selected_crop:
            crop_id = int(selected_crop.split()[-1]) - 1
            selected_crop_data = crop_health[crop_id]
            st.write(f"### Details for {selected_crop}:")
            st.write(f"- **Health Status:** {selected_crop_data['Health Status']}")
