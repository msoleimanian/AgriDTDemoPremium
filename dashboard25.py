import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt


# Helper function for image processing and crop health detection
def detect_crops_and_health_week2(image, min_pixels=900):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    closed_green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_green_mask, connectivity=8)

    min_component_area = 500

    processed_image = image.copy()
    crop_health = []
    total_crops = 0
    healthy_crops = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_component_area:
            total_crops += 1
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[
                i, cv2.CC_STAT_HEIGHT]

            crop_mask = closed_green_mask[y:y + h, x:x + w]
            green_pixels = cv2.countNonZero(crop_mask)
            total_pixels = w * h
            green_percentage = (green_pixels / total_pixels) * 100

            if green_percentage >= 40 and green_pixels >= min_pixels:
                health_status = "Healthy"
                healthy_crops += 1
                border_color = (0, 255, 0)  # Green for healthy crops
            else:
                health_status = "Unhealthy"
                border_color = (255, 0, 0)  # Red for unhealthy crops

            cv2.rectangle(processed_image, (x, y), (x + w, y + h), border_color, 3)
            cv2.putText(processed_image, f"Crop {total_crops}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        border_color, 2)

            crop_health.append({
                "Crop": total_crops,
                "Health Status": health_status,
                "Green Pixels": green_pixels,
                "Total Pixels": total_pixels,
                "Green Percentage": green_percentage
            })

    improved_mask = np.ones_like(image) * 255
    improved_mask[closed_green_mask > 0] = [0, 255, 0]

    return processed_image, crop_health, healthy_crops, total_crops, improved_mask


# Function to plot 2D version of the area (with clickable crops)
def plot_2d_area():
    import plotly.graph_objects as go
    import streamlit as st

    # Example crop health data with updated values
    crop_health = [
        {"Crop": 1, "Health Status": "Healthy", "Green Percentage": 70, "Predicted Canopy": 1.5, "Plant Height": 0.22,
         "Longest Leaf": 0.17, "Yield": 160},
        {"Crop": 2, "Health Status": "Healthy", "Green Percentage": 85, "Predicted Canopy": 1.6, "Plant Height": 0.23,
         "Longest Leaf": 0.18, "Yield": 180},
        {"Crop": 3, "Health Status": "Unhealthy", "Green Percentage": 30, "Predicted Canopy": 1.1, "Plant Height": 0.19,
         "Longest Leaf": 0.16, "Yield": 120},
        {"Crop": 4, "Health Status": "Healthy", "Green Percentage": 95, "Predicted Canopy": 1.8, "Plant Height": 0.22,
         "Longest Leaf": 0.19, "Yield": 200},
        {"Crop": 5, "Health Status": "Healthy", "Green Percentage": 60, "Predicted Canopy": 1.4, "Plant Height": 0.21,
         "Longest Leaf": 0.16, "Yield": 175},
        {"Crop": 6, "Health Status": "Unhealthy", "Green Percentage": 40, "Predicted Canopy": 1.2, "Plant Height": 0.20,
         "Longest Leaf": 0.15, "Yield": 130},
        {"Crop": 7, "Health Status": "Healthy", "Green Percentage": 75, "Predicted Canopy": 1.7, "Plant Height": 0.23,
         "Longest Leaf": 0.18, "Yield": 190},
        {"Crop": 8, "Health Status": "Unhealthy", "Green Percentage": 20, "Predicted Canopy": 1.0, "Plant Height": 0.19,
         "Longest Leaf": 0.15, "Yield": 125},
    ]

    def plot_3d_crops(crop_health, selected_crop):
        """
        Visualizes crop health using 3D lines with spheres on top.

        Parameters:
            crop_health (list): List of dictionaries with crop health data.
            selected_crop (int): Index of the selected crop (1-based).
        """
        # Ensure crop_health contains 8 items (add placeholders if necessary)
        while len(crop_health) < 8:
            crop_health.append({
                "Crop": len(crop_health) + 1,
                "Health Status": "Not Detected",
                "Green Pixels": 0,
                "Total Pixels": 0,
                "Green Percentage": 0
            })

        # Initialize 3D plot
        fig = go.Figure()

        # Define 3D positions for the crops (spread on the X-Y plane)
        x_positions = [0, 1, 2, 3, 0, 1, 2, 3]
        y_positions = [0, 0, 0, 0, 1, 1, 1, 1]
        z_positions = [0] * 8  # Base height of lines (z = 0)

        # Add each crop as a line with a spherical head
        for i, crop in enumerate(crop_health):
            # Line base (z = 0) to crop height (z = Green Percentage or default)
            green_percentage = crop["Green Percentage"]
            z_height = max(green_percentage / 100, 0.2)  # Minimum height of 0.2 for visibility

            x = [x_positions[i], x_positions[i]]
            y = [y_positions[i], y_positions[i]]
            z = [0, z_height]

            # Line and sphere color based on green percentage (gradient)
            color = f"rgb({255 - int(2.55 * green_percentage)}, {int(2.55 * green_percentage)}, 0)"  # Gradient green to red

            # Highlight selected crop with a yellow sphere
            sphere_color = 'yellow' if selected_crop == i + 1 else color

            # Larger sphere size
            sphere_size = 20 + (green_percentage / 5)  # Minimum size 20, scales with percentage

            # Add the line to the plot
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color=color, width=5),
                name=f"Crop {crop['Crop']}",
                showlegend=False  # Hide legend for individual lines
            ))

            # Add the sphere at the top of the line
            fig.add_trace(go.Scatter3d(
                x=[x[1]], y=[y[1]], z=[z[1]],  # Sphere position at the top of the line
                mode='markers+text',
                marker=dict(size=sphere_size, color=sphere_color, opacity=0.9),
                text=f"Crop {crop['Crop']}<br>{green_percentage}% Healthy<br>Status: {crop['Health Status']}",
                textposition='top center',
                name=f"Crop {crop['Crop']}"
            ))

        # Set layout for the 3D plot
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X-axis', backgroundcolor="rgb(230, 230, 230)"),
                yaxis=dict(title='Y-axis', backgroundcolor="rgb(230, 230, 230)"),
                zaxis=dict(title='Health (%)', range=[0, 1], backgroundcolor="rgb(240, 240, 240)"),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            title="3D Crop Health Visualization",
            showlegend=False
        )

        # Display the 3D plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Enhanced 3D Crop Health Visualization")

        # Select a crop to highlight
        selected_crop_info = st.selectbox("Select a Crop to Highlight", range(1, 9))

        # Plot the 3D crops
        plot_3d_crops(crop_health, selected_crop_info)

    with col2:
        # Display crop details for the selected crop
        if selected_crop_info:
            selected_crop_data = next((crop for crop in crop_health if crop["Crop"] == selected_crop_info), None)

            if selected_crop_data:
                st.subheader(f"Details for Crop {selected_crop_info}")
                st.write(f"**Health Status:** {selected_crop_data['Health Status']}")
                st.write(f"**Predicted Canopy Size:** {selected_crop_data['Predicted Canopy']} m²")
                st.write(f"**Predicted Plant Height:** {selected_crop_data['Plant Height']} m")
                st.write(f"**Predicted Longest Leaf:** {selected_crop_data['Longest Leaf']} m")
                st.write(f"**Predicted Yield in Harvest:** {selected_crop_data['Yield']} g")


# Streamlit app for selecting areas and displaying images and results
st.set_page_config(page_title="Crop Health Dashboard - Week 2", layout="wide")
st.title("Crop Health Detection Dashboard - Week 2")

# Directory with images (Ensure this folder contains your images)
image_folder = "./Images"  # Ensure this folder contains your images
image_files = [f"{i}.jpeg" for i in range(1, 20)]  # Images from 1.jpeg to 19.jpeg

# Dropdown for selecting the area
area_selected = st.selectbox("Select Area to View Image Processing", [f"Area {i}" for i in range(1, 20)])

# Get the corresponding image for the selected area
area_index = int(area_selected.split()[1]) - 1  # Extract index of the selected area
current_image_file = image_files[area_index]
image_path = os.path.join(image_folder, current_image_file)

# Load the image from the folder based on the selected area
image = np.array(Image.open(image_path))

# Detect crops and calculate health for Week 2
processed_image, crop_health, healthy_crops, total_crops, improved_mask = detect_crops_and_health_week2(image)

# Create a two-column layout for the images
img_col1, img_col2 = st.columns([1, 2])

# Display Processed Image on the left side
with img_col1:
    st.image(processed_image, caption="Processed Image", use_container_width=True)

# Create a column for the right side summary and crop details
with img_col2:
    # Overall Health Status Box
    st.subheader("Area Health Status")

    # Calculate the percentage of healthy crops
    healthy_percentage = (healthy_crops / total_crops) * 100 if total_crops > 0 else 0

    # Set the desired yield to 250 grams for each area
    desired_yield = 250  # Fixed yield for each area

    # Show the box with health status and yield
    st.markdown(f"""
    **Health Status of Area:**
    - Healthy Crops: {healthy_crops}/{total_crops} ({healthy_percentage:.2f}%) 
    - Desired Yield for the Area: {desired_yield} grams
    """)

    # Crop Health Status Table (Three columns layout)
    st.subheader("Crop Health Status")

    # Split crop_health into three columns for a better view
    cols = st.columns(3)

    # Distribute crops across the three columns
    for i in range(3):
        with cols[i]:
            # Calculate the range of crops to display in this column
            start_index = i * len(crop_health) // 3
            end_index = (i + 1) * len(crop_health) // 3 if i != 2 else len(crop_health)

            # Create a DataFrame for this column's crops
            crop_column_data = {
                "Crop": [crop["Crop"] for crop in crop_health[start_index:end_index]],
                "Health Status": [crop["Health Status"] for crop in crop_health[start_index:end_index]]
            }

            crop_column_df = pd.DataFrame(crop_column_data)
            st.table(crop_column_df)


# Display the 2D area visualization with the selected crop highlighted
plot_2d_area()

