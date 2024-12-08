import streamlit as st
import time
import random

import plotly.graph_objects as go


def plot_3d_crops(crop_health, selected_crop):
    """
    Visualizes crop health using 3D lines with spheres on top.

    Parameters:
        crop_health (list): List of dictionaries with crop health data.
        selected_crop (int): Index of the selected crop (1-based).
    """
    # Ensure crop_health contains 10 items (add placeholders if necessary)
    while len(crop_health) < 10:
        crop_health.append({
            "Crop": len(crop_health) + 1,
            "Health Status": "Not Detected",
            "Green Pixels": 0,
            "Total Pixels": 0,
            "Green Percentage": 0
        })

    # Initialize 3D plot
    fig = go.Figure()

    # Define 3D positions for the crops (spread on the X-Y plane for 10 crops)
    x_positions = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    y_positions = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    z_positions = [0] * 10  # Base height of lines (z = 0)

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
        showlegend=False
    )

    # Display the 3D plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)



def plot_3d_crops2(crop_health, selected_crop, chart_key):
    """
    Visualizes crop health using 3D lines with spheres on top.

    Parameters:
        crop_health (list): List of dictionaries with crop health data.
        selected_crop (int): Index of the selected crop (1-based).
        chart_key (str): A unique key for the Streamlit chart element.
    """
    # Ensure crop_health contains 10 items (add placeholders if necessary)
    while len(crop_health) < 10:
        crop_health.append({
            "Crop": len(crop_health) + 1,
            "Health Status": "Not Detected",
            "Green Pixels": 0,
            "Total Pixels": 0,
            "Green Percentage": 0
        })

    # Initialize 3D plot
    fig = go.Figure()

    # Define 3D positions for the crops (spread on the X-Y plane for 10 crops)
    x_positions = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    y_positions = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    z_positions = [0] * 10  # Base height of lines (z = 0)

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
        showlegend=False
    )

    # Display the 3D plot in Streamlit with a unique key
    st.plotly_chart(fig, use_container_width=True, key=chart_key)

st.set_page_config(page_title="Monitoring Dashboard", layout="wide")

selected_farm = st.sidebar.selectbox("Select the farm:", ["Pak Choy1", "Pak Choy2"])

if selected_farm == 'Pak Choy1':
    # Function to create a styled card
    def create_card(title, value, unit, color):
        card_html = f"""
        <div style="background-color: {color}; padding: 8px; border-radius: 6px; width: 120px; height: 80px; text-align: center; box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);">
            <h5 style="margin: 0; color: #fff; font-size: 12px;">{title}</h5>
            <h3 style="margin: 5px 0; color: #fff; font-size: 16px;">{value} {unit}</h3>
        </div>
        """
        return card_html

    # Function to create a circular progress bar
    def circular_progress_bar(percentage, label, dynamic_label, health_category):
        # Determine the color based on health status
        health_colors = {
            "Poor": "#FF0000",     # Red
            "Weak": "#FFA500",     # Orange
            "Normal": "#FFFF00",   # Yellow
            "Good": "#4CAF50",     # Green
            "Excellent": "#008000" # Dark Green
        }
        color = health_colors.get(health_category, "#DDDDDD")  # Default to gray if category is not found

        progress_html = f"""
        <style>
        @keyframes progress-anim {{
            0% {{
                background: conic-gradient(
                    {color} 0deg,
                    #ddd 0deg
                );
            }}
            100% {{
                background: conic-gradient(
                    {color} {percentage * 3.6}deg,
                    #ddd {percentage * 3.6}deg
                );
            }}
        }}
        .progress-container {{
            position: relative;
            width: 150px;
            height: 150px;
            border-radius: 50%;
            background: conic-gradient(#ddd 0deg, #ddd 360deg);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            color: #000;
            animation: progress-anim 2s ease-out forwards;
        }}
        .progress-inner {{
            position: absolute;
            width: 120px;
            height: 120px;
            background: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: bold;
        }}
        .progress-label {{
            text-align: center;
            font-size: 16px;
            margin-top: 10px;
        }}
        </style>
        <div>
            <div class="progress-container">
                <div class="progress-inner">
                    <div>{dynamic_label}</div>
                </div>
            </div>
            <p class="progress-label">{label}</p>
        </div>
        """
        return progress_html



    # Fixed Plant Growth Monitoring Data
    health_status = 70  # Good = 70%
    pak_choy_progress = 50  # Representing 2 weeks
    desired_yield_percentage = 1.4 / 1.7 * 100  # 1.4 KG out of 1.7 KG target

    # Display the static Plant Growth Monitoring Section
    st.title("Monitoring")
    st.write("### Plant Growth Monitoring")

    with st.expander("Plant Growth Monitoring", expanded=True):
        import streamlit as st


        # Function for circular progress bar
        def circular_progress_bar(percentage, label, dynamic_label, color):
            progress_html = f"""
            <div style="position: relative; width: 150px; height: 150px; border-radius: 50%; background: conic-gradient(
                {color} {percentage * 3.6}deg,
                #ddd {percentage * 3.6}deg
            ); display: flex; align-items: center; justify-content: center; font-size: 20px; color: #000;">
                <div style="position: absolute; width: 120px; height: 120px; background: white; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                    <div>{dynamic_label}</div>
                </div>
            </div>
            <p style="text-align: center; font-size: 16px;">{label}</p>
            """
            return progress_html


        # Function for line progress bar
        def line_progress_bar(percentage, label, dynamic_label, health_category, max_value):
            # Determine the color based on health status
            health_colors = {
                "Poor": "#FF0000",  # Red
                "Weak": "#FFA500",  # Orange
                "Normal": "#FFFF00",  # Yellow
                "Good": "#4CAF50",  # Green
                "Excellent": "#008000"  # Dark Green
            }
            color = health_colors.get(health_category, "#DDDDDD")  # Default to gray if category is not found

            progress_html = f"""
            <style>
            .progress-container {{
                width: 100%;
                background-color: #ddd;
                height: 20px;
                border-radius: 10px;
                margin: 10px 0;
            }}
            .progress-bar {{
                width: {percentage}%;
                background-color: {color};
                height: 100%;
                border-radius: 10px;
                text-align: center;
                line-height: 20px;
                font-weight: bold;
                color: white;
            }}
            .progress-label {{
                text-align: center;
                font-size: 16px;
                color: #555;
                font-weight: bold;
            }}
            </style>
            <div>
                <div class="progress-container">
                    <div class="progress-bar">
                        {dynamic_label}
                    </div>
                </div>
                <p class="progress-label">{label}</p>
            </div>
            """
            return progress_html


        # Simulate health status and other metrics
        health_status = 75  # Health Status (Good)
        pak_choy_week = 1  # Week 1 selected, corresponds to 25% progress
        desired_yield_percentage = (1.4 / 1.7) * 100  # Desired yield as a percentage (1.4KG out of 1.7KG)

        # For Pak Choy Growth, we set 25% progress for Week 1
        pak_choy_progress = 25  # Fixed 25% progress for Week 1


        # Function to assign health category based on the health status percentage
        def get_health_category(percentage):
            if percentage < 20:
                return "Poor"
            elif percentage < 40:
                return "Weak"
            elif percentage < 60:
                return "Normal"
            elif percentage < 80:
                return "Good"
            else:
                return "Excellent"


        # Get the health category
        health_category = get_health_category(health_status)

        # Display the progress bars
        # Create the layout with three columns
        col1, col2, col3 = st.columns(3)

        # Health Circular Progress Bar
        with col1:
            st.markdown(
                circular_progress_bar(health_status, "Current Health", "Good", "#4CAF50"),  # Green for Good health
                unsafe_allow_html=True,
            )

        # Pak Choy Growth Line Progress Bar
        with col3:
            import streamlit as st


            # Function to display pest detection card
            def pest_detection_card(detected: bool):
                # Set the card's background color based on pest detection
                color = "#FF5722" if detected else "#4CAF50"  # Red for detected, Green for not detected
                status_text = "Pest Detected" if detected else "No Pest Detected"
                emoji = "🐞" if detected else "✅"  # Pest emoji for detection, Check mark for no pest

                # HTML for the card
                card_html = f"""
                <div style="background-color:{color}; color:white; padding: 20px; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold;">
                    <p>{emoji}</p>
                    <p>{status_text}</p>
                </div>
                """
                return card_html


            # Simulating pest detection state (True for detected, False for not detected)
            pest_detected = False  # Change to True to simulate pest detection

            # Display the pest detection card
            st.markdown(pest_detection_card(pest_detected), unsafe_allow_html=True)

        # Desired Yield Circular Progress Bar
        with col2:
            st.markdown(
                circular_progress_bar(desired_yield_percentage, "Estimated Yield(The target yield is 1.7 KG, 82% of which will be achieved.)", "1.4 KG", "#FFA500"),
                # Orange for desired yield
                unsafe_allow_html=True,
            )

        # Line progress bar for Pak Choy Growth (Week 1)
        st.markdown(
            line_progress_bar(pak_choy_progress, f"Pak Choy Growth (Week {pak_choy_week})", f"{pak_choy_week} Week",
                              get_health_category(pak_choy_progress), 100),
            unsafe_allow_html=True,
        )

    # Create a placeholder for the real-time Environmental Parameters
    st.write("### Environmental Parameters")
    placeholder = st.empty()


    data = {
        "ph_level": round(random.uniform(6.0, 7.5), 2),
        "ec_level": round(random.uniform(1.0, 2.0), 2),
        "temperature": round(random.uniform(20.0, 30.0), 1),
        "tds_level": random.randint(400, 600),
        "orp_level": random.randint(200, 400),
    }

    # Update the placeholder with new Environmental Parameters
    with placeholder.container():
        with st.expander("Environmental Parameters", expanded=True):  # Add an expander for the cards
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.markdown(create_card("pH Level", data["ph_level"], "", "#4CAF50"), unsafe_allow_html=True)

            with col2:
                st.markdown(create_card("EC Level", data["ec_level"], "mS/cm", "#2196F3"), unsafe_allow_html=True)

            with col3:
                st.markdown(create_card("Temperature", data["temperature"], "°C", "#FF9800"), unsafe_allow_html=True)

            with col4:
                st.markdown(create_card("TDS", data["tds_level"], "ppm", "#673AB7"), unsafe_allow_html=True)

            with col5:
                st.markdown(create_card("ORP", data["orp_level"], "mV", "#FF5722"), unsafe_allow_html=True)

            st.write('')


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
            st.subheader("Predicting Crop Health")

            # Select a crop to highlight
            selected_crop_info = st.selectbox("Select a Crop to Highlight", range(1, 9))

            # Plot the 3D crops
            plot_3d_crops(crop_health, selected_crop_info)

        with col2:
            # Display crop details for the selected crop
            if selected_crop_info:
                selected_crop_data = next((crop for crop in crop_health if crop["Crop"] == selected_crop_info), None)

                if selected_crop_data:
                    # HTML content for displaying crop details
                    html_content = f"""
                    <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 15px; background-color: #f9f9f9;">
                        <h4 style="color: #4CAF50; text-align: center;">Crop Information</h4>
                        <p><b>Health Status:</b> {selected_crop_data['Health Status']}</p>
                        <p><b>Green Percentage:</b> {selected_crop_data['Green Percentage']}%</p>
                        <p><b>Predicted Canopy Size:</b> {selected_crop_data['Predicted Canopy']} m²</p>
                        <p><b>Predicted Plant Height:</b> {selected_crop_data['Plant Height']} m</p>
                        <p><b>Predicted Longest Leaf:</b> {selected_crop_data['Longest Leaf']} m</p>
                        <p><b>Predicted Yield in Harvest:</b> {selected_crop_data['Yield']} g</p>
                    </div>
                    """

                    # Render the HTML content in Streamlit
                    st.markdown(html_content, unsafe_allow_html=True)


    import streamlit as st
    import random
    import pandas as pd

    # --- Inputs and Calculations ---
    # Define the target yield for comparison
    target_yield = 950  # grams

    # Generate random estimated yields for 19 areas
    areas = [f"Area {i + 1}" for i in range(1, 20)]
    area_yields = [random.uniform(500, 700) for _ in range(19)]

    # Create a DataFrame for area-wise reporting
    area_data = pd.DataFrame({
        "Area": areas,
        "Estimated Yield (grams)": area_yields
    })

    # Add a column to classify yields as "Good" or "Needs Improvement"
    area_data["Yield Status"] = area_data["Estimated Yield (grams)"].apply(
        lambda x: "Good" if x >= (target_yield * 0.8) else "Needs Improvement"
    )

    # --- Streamlit App Interface ---
    # Expander for Area Report
    with st.expander("Click to view Area-wise Analyze for 19 Areas"):
        st.markdown("<h4 style='text-align: center; color: #4CAF50;'>Area-wise Yield Analyze</h4>",
                    unsafe_allow_html=True)

        # Display the area report table
        st.table(area_data)

        # Explanation Section
        st.markdown("""
        <div style="border: 2px dashed #4CAF50; border-radius: 10px; padding: 15px; background-color: #fff;">
            <h4 style="color: #4CAF50; text-align: center;">Explanation</h4>
            <p><b>Good Yield:</b> Areas where the estimated yield is at least 80% of the target yield (760 grams).</p>
            <p><b>Needs Improvement:</b> Areas where the estimated yield is less than 80% of the target yield.</p>
            <p>This classification helps identify areas that need attention to improve yields and ensure consistent performance.</p>
        </div>
        """, unsafe_allow_html=True)

    # Streamlit app for selecting areas and displaying images and results
    st.subheader("Crop Health Analyze")

    # Directory with images (Ensure this folder contains your images)
    image_folder = "./Images"  # Ensure this folder contains your images
    image_files = [f"{i}.jpeg" for i in range(1, 20)]  # Images from 1.jpeg to 19.jpeg


    # Example function to simulate the health status of each area
    def get_area_status(area):
        # For simplicity, assume area 1 is fine and areas 2-19 need premium
        if area == "Area 1":
            return "OK"
        else:
            return "Needs Premium"

    colarea, colcrop = st.columns(2)
    # Dropdown for selecting the area
    with colarea:
        area_selected = st.selectbox("Select Area to View Image Processing", [f"Area {i}" for i in range(1, 20)])
    with colcrop:
        selected_crop_info = st.selectbox("Select a Crop to Highlight", range(1, 9))

    # Get the status of the selected area
    area_status = get_area_status(area_selected)

    # Show results
    if area_status == "Needs Premium":
        st.markdown("""
            <div style="font-size:18px; color: red; border: 2px solid #FF5733; padding: 15px; border-radius: 10px; background-color: #f9f9f9;">
                <p style="text-align: center;"><strong>Upgrade to Premium Edition for full functionality.</strong></p>
                <p style="text-align: center;">Contact <a href="mailto:nurfadhlina@upm.edu.my">nurfadhlina@upm.edu.my</a> for further details.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Display the image or results for area 1 (if area is fine)

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
            st.subheader("Current")
            st.write('')
            st.image(processed_image, caption="Processed Image", use_container_width=True)

        # Create a column for the right side summary and crop details
        with img_col2:

            st.subheader("Predicted")

            # Select a crop to highlight

            # Plot the 3D crops
            plot_3d_crops(crop_health, selected_crop_info)

    # Overall Health Status Box
    import streamlit as st

    # Sample Data (replace with your variables)
    healthy_crops = 80
    total_crops = 100

    # Calculate the percentage of healthy crops
    healthy_percentage = (healthy_crops / total_crops) * 100 if total_crops > 0 else 0

    # Set the desired yield to 250 grams for each area
    desired_yield = 250  # Fixed yield for each area

    import streamlit as st
    import random
    import pandas as pd

    # --- Inputs and Calculations ---
    # Define the target yield and estimated yield for the area
    target_yield = 950  # grams
    estimated_yield = 730  # grams

    # Calculate the percentage of the target yield achieved
    percentage_gain = (estimated_yield / target_yield) * 100

    # Generate mock data for 8 crops (randomized for demonstration purposes)
    # Crop names
    crops = [f"Crop {i + 1}" for i in range(8)]

    # Weekly canopy coverage percentages and predicted yields
    # Week 1 canopy is the observed data, and subsequent weeks are predictions
    week1_canopy = [random.uniform(60, 80) for _ in range(8)]
    predicted_week2 = [random.uniform(70, 85) for _ in range(8)]
    predicted_week3 = [random.uniform(75, 88) for _ in range(8)]
    predicted_week4 = [random.uniform(80, 90) for _ in range(8)]
    predicted_yield = [random.uniform(80, 90) for _ in range(8)]  # Predicted yield in grams

    # Create a DataFrame for displaying data in a table
    crop_data = pd.DataFrame({
        "Crop": crops,
        "Week 1 Canopy (%)": week1_canopy,
        "Predicted Week 2 (%)": predicted_week2,
        "Predicted Week 3 (%)": predicted_week3,
        "Predicted Week 4 (%)": predicted_week4,
        "Predicted Yield (grams)": predicted_yield,
    })

    # --- HTML Content for Display ---
    # Combine Health Status and Explanation in one section
    html_content = f"""
    <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; background-color: #f9f9f9;">
        <h4 style="color: #4CAF50; text-align: center;">Health Status and Explanation</h4>
        <p><b>Target Yield for the Area:</b> {target_yield} grams</p>
        <div style="font-size:18px; color: red; border: 2px solid #FF5733; padding: 15px; border-radius: 10px; background-color: #f9f9f9;">
        <p style="text-align: center;"><strong>Upgrade to Premium Edition for full functionality.</strong></p>
        <p style="text-align: center;">Get access to the following premium features:</p>
        <ul style="font-size:16px;">
            <li><strong>Estimated Yield for the Area</strong>: Get precise yield predictions for different areas of your crop system.</li>
            <li><strong>Percentage of Target Achieved</strong>: Measure and track your crop's performance against target goals.</li>
        </ul>
        <p style="text-align: center;">Contact <a href="mailto:nurfadhlina@upm.edu.my">nurfadhlina@upm.edu.my</a> for further details.</p>
    </div>
        <hr style="border-top: 1px solid #4CAF50;">
        <h4 style="color: #4CAF50;">Explanation</h4>
        <p><b>Target Yield:</b> This represents the ideal yield for the area (in grams).</p>
        <p><b>Estimated Yield:</b> This is the current predicted yield based on observed data.</p>
        <p><b>Percentage of Target Achieved:</b> Shows how close the estimated yield is to the target yield.</p>
        <p><b>Crop Table:</b> Provides weekly canopy data and predicted yields for 8 crops. The values are randomized for this demonstration.</p>
    </div>
    """

    # --- Streamlit App Interface ---
    # Display the combined HTML content in Streamlit
    st.markdown(html_content, unsafe_allow_html=True)

    # Button to display the table
    if st.button("Show Crop Canopy & Yield Predictions"):
        st.markdown("""
                <div style="font-size:18px; color: red; border: 2px solid #FF5733; padding: 15px; border-radius: 10px; background-color: #f9f9f9;">
                    <p style="text-align: center;"><strong>Upgrade to Premium Edition for full functionality.</strong></p>
                    <p style="text-align: center;">Contact <a href="mailto:nurfadhlina@upm.edu.my">nurfadhlina@upm.edu.my</a> for further details.</p>
                </div>
            """, unsafe_allow_html=True)

    # Crop Health Status Table (Three columns layout)
    import streamlit as st

    # Define recommendations for 19 areas
    recommendations = [
        {
            "Area": f"Area {i + 1}",
            "Weekly Recommendation": {
                "Week 2": f"Solution A: {250 + i * 10} mml + Solution B: {350 - i * 5} mml to adjust pH and EC.",
                "Action": "Crop 5 should be cut and replaced." if i % 3 == 0 else "No crops need replacing.",
                "Pests": "Pests not detected." if i % 2 == 0 else "Pests detected! Apply pest control measures.",
            },
            "Future Recommendations": {
                "Week 3": {
                    "Solution A": f"{200 + i * 5} mml",
                    "Solution B": f"{300 - i * 10} mml",
                    "Instructions": "Adjust Solution B for stable EC levels.",
                },
                "Week 4": {
                    "Solution A": f"{150 + i * 2} mml",
                    "Solution B": f"{250 - i * 7} mml",
                    "Instructions": "Maintain pH using Solution A only.",
                },
            },
        }
        for i in range(19)
    ]

    # Dropdown for selecting an area
    selected_area_name = area_selected

    # Find the selected area data
    selected_area = next(area for area in recommendations if area["Area"] == selected_area_name)

    # Display selected area recommendations
    st.markdown(
        f"""
            <div class="recommendation-box">
                <h4 style="color: #4CAF50; text-align: center;">Recommendations {selected_area['Area']}</h4>
                <div style="font-size:18px; color: red; border: 2px solid #FF5733; padding: 15px; border-radius: 10px; background-color: #f9f9f9;">
            <p style="text-align: center;"><strong>Upgrade to Premium Edition for full functionality.</strong></p>
            <p style="text-align: center;">Contact <a href="mailto:nurfadhlina@upm.edu.my">nurfadhlina@upm.edu.my</a> for further details.</p>
        </div>
            </div>
            """,
        unsafe_allow_html=True,
    )

    # Toggle for More Info
    if st.button("📖 Show More Info"):
        st.markdown("""
                <div style="font-size:18px; color: red; border: 2px solid #FF5733; padding: 15px; border-radius: 10px; background-color: #f9f9f9;">
                    <p style="text-align: center;"><strong>Upgrade to Premium Edition for full functionality.</strong></p>
                    <p style="text-align: center;">Contact <a href="mailto:nurfadhlina@upm.edu.my">nurfadhlina@upm.edu.my</a> for further details.</p>
                </div>
            """, unsafe_allow_html=True)


if selected_farm == 'Pak Choy2':
    # Function to create a styled card
    def create_card(title, value, unit, color):
        card_html = f"""
        <div style="background-color: {color}; padding: 8px; border-radius: 6px; width: 120px; height: 80px; text-align: center; box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);">
            <h5 style="margin: 0; color: #fff; font-size: 12px;">{title}</h5>
            <h3 style="margin: 5px 0; color: #fff; font-size: 16px;">{value} {unit}</h3>
        </div>
        """
        return card_html

    # Function to create a circular progress bar
    def circular_progress_bar(percentage, label, dynamic_label, health_category):
        # Determine the color based on health status
        health_colors = {
            "Poor": "#FF0000",     # Red
            "Weak": "#FFA500",     # Orange
            "Normal": "#FFFF00",   # Yellow
            "Good": "#4CAF50",     # Green
            "Excellent": "#008000" # Dark Green
        }
        color = health_colors.get(health_category, "#DDDDDD")  # Default to gray if category is not found

        progress_html = f"""
        <style>
        @keyframes progress-anim {{
            0% {{
                background: conic-gradient(
                    {color} 0deg,
                    #ddd 0deg
                );
            }}
            100% {{
                background: conic-gradient(
                    {color} {percentage * 3.6}deg,
                    #ddd {percentage * 3.6}deg
                );
            }}
        }}
        .progress-container {{
            position: relative;
            width: 150px;
            height: 150px;
            border-radius: 50%;
            background: conic-gradient(#ddd 0deg, #ddd 360deg);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            color: #000;
            animation: progress-anim 2s ease-out forwards;
        }}
        .progress-inner {{
            position: absolute;
            width: 120px;
            height: 120px;
            background: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: bold;
        }}
        .progress-label {{
            text-align: center;
            font-size: 16px;
            margin-top: 10px;
        }}
        </style>
        <div>
            <div class="progress-container">
                <div class="progress-inner">
                    <div>{dynamic_label}</div>
                </div>
            </div>
            <p class="progress-label">{label}</p>
        </div>
        """
        return progress_html



    # Fixed Plant Growth Monitoring Data
    health_status = 70  # Good = 70%
    pak_choy_progress = 50  # Representing 2 weeks
    desired_yield_percentage = 1.4 / 1.7 * 100  # 1.4 KG out of 1.7 KG target

    # Display the static Plant Growth Monitoring Section
    st.title("Monitoring")
    st.write("### Plant Growth Monitoring")

    with st.expander("Plant Growth Monitoring", expanded=True):
        import streamlit as st


        # Function for circular progress bar
        def circular_progress_bar(percentage, label, dynamic_label, color):
            progress_html = f"""
            <div style="position: relative; width: 150px; height: 150px; border-radius: 50%; background: conic-gradient(
                {color} {percentage * 3.6}deg,
                #ddd {percentage * 3.6}deg
            ); display: flex; align-items: center; justify-content: center; font-size: 20px; color: #000;">
                <div style="position: absolute; width: 120px; height: 120px; background: white; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                    <div>{dynamic_label}</div>
                </div>
            </div>
            <p style="text-align: center; font-size: 16px;">{label}</p>
            """
            return progress_html


        # Function for line progress bar
        def line_progress_bar(percentage, label, dynamic_label, health_category, max_value):
            # Determine the color based on health status
            health_colors = {
                "Poor": "#FF0000",  # Red
                "Weak": "#FFA500",  # Orange
                "Normal": "#FFFF00",  # Yellow
                "Good": "#4CAF50",  # Green
                "Excellent": "#008000"  # Dark Green
            }
            color = health_colors.get(health_category, "#DDDDDD")  # Default to gray if category is not found

            progress_html = f"""
            <style>
            .progress-container {{
                width: 100%;
                background-color: #ddd;
                height: 20px;
                border-radius: 10px;
                margin: 10px 0;
            }}
            .progress-bar {{
                width: 50%;
                background-color: BLUE;
                height: 100%;
                border-radius: 10px;
                text-align: center;
                line-height: 20px;
                font-weight: bold;
                color: white;
            }}
            .progress-label {{
                text-align: center;
                font-size: 16px;
                color: #555;
                font-weight: bold;
            }}
            </style>
            <div>
                <div class="progress-container">
                    <div class="progress-bar">
                        {dynamic_label}
                    </div>
                </div>
                <p class="progress-label">{label}</p>
            </div>
            """
            return progress_html


        # Simulate health status and other metrics
        health_status = 75  # Health Status (Good)
        pak_choy_week = 2  # Week 1 selected, corresponds to 25% progress
        desired_yield_percentage = (1.4 / 1.7) * 100  # Desired yield as a percentage (1.4KG out of 1.7KG)

        # For Pak Choy Growth, we set 25% progress for Week 1
        pak_choy_progress = 25  # Fixed 25% progress for Week 1


        # Function to assign health category based on the health status percentage
        def get_health_category(percentage):
            if percentage < 20:
                return "Poor"
            elif percentage < 40:
                return "Weak"
            elif percentage < 60:
                return "Normal"
            elif percentage < 80:
                return "Good"
            else:
                return "Excellent"


        # Get the health category
        health_category = get_health_category(health_status)

        # Display the progress bars
        # Create the layout with three columns
        col1, col2, col3 = st.columns(3)

        # Health Circular Progress Bar
        with col1:
            st.markdown(
                circular_progress_bar(health_status, "Health", "Good", "#4CAF50"),  # Green for Good health
                unsafe_allow_html=True,
            )

        # Pak Choy Growth Line Progress Bar
        with col3:
            import streamlit as st


            # Function to display pest detection card
            def pest_detection_card(detected: bool):
                # Set the card's background color based on pest detection
                color = "#FF5722" if detected else "#4CAF50"  # Red for detected, Green for not detected
                status_text = "Pest Detected" if detected else "No Pest Detected"
                emoji = "🐞" if detected else "✅"  # Pest emoji for detection, Check mark for no pest

                # HTML for the card
                card_html = f"""
                <div style="background-color:{color}; color:white; padding: 20px; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold;">
                    <p>{emoji}</p>
                    <p>{status_text}</p>
                </div>
                """
                return card_html


            # Simulating pest detection state (True for detected, False for not detected)
            pest_detected = False  # Change to True to simulate pest detection

            # Display the pest detection card
            st.markdown(pest_detection_card(pest_detected), unsafe_allow_html=True)

        # Desired Yield Circular Progress Bar
        with col2:
            st.markdown(
                circular_progress_bar(desired_yield_percentage, "Estimated Yield(The target yield is 1.7 KG, 82% of which will be achieved.)", "1.4 KG", "#FFA500"),
                # Orange for desired yield
                unsafe_allow_html=True,
            )

        # Line progress bar for Pak Choy Growth (Week 1)
        st.markdown(
            line_progress_bar(pak_choy_progress, f"Pak Choy Growth (Week {pak_choy_week})", f"{pak_choy_week} Week",
                              get_health_category(pak_choy_progress), 100),
            unsafe_allow_html=True,
        )

    # Create a placeholder for the real-time Environmental Parameters
    st.write("### Environmental Parameters")
    placeholder = st.empty()


    data = {
        "ph_level": round(random.uniform(6.0, 7.5), 2),
        "ec_level": round(random.uniform(1.0, 2.0), 2),
        "temperature": round(random.uniform(20.0, 30.0), 1),
        "tds_level": random.randint(400, 600),
        "orp_level": random.randint(200, 400),
    }

    # Update the placeholder with new Environmental Parameters
    with placeholder.container():
        with st.expander("Environmental Parameters", expanded=True):  # Add an expander for the cards
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.markdown(create_card("pH Level", data["ph_level"], "", "#4CAF50"), unsafe_allow_html=True)

            with col2:
                st.markdown(create_card("EC Level", data["ec_level"], "mS/cm", "#2196F3"), unsafe_allow_html=True)

            with col3:
                st.markdown(create_card("Temperature", data["temperature"], "°C", "#FF9800"), unsafe_allow_html=True)

            with col4:
                st.markdown(create_card("TDS", data["tds_level"], "ppm", "#673AB7"), unsafe_allow_html=True)

            with col5:
                st.markdown(create_card("ORP", data["orp_level"], "mV", "#FF5722"), unsafe_allow_html=True)

            st.write('')


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
    def plot_2d_areaArea1():
        import plotly.graph_objects as go
        import streamlit as st

        # Example crop health data with updated values
        crop_health = [
            {"Crop": 1, "Health Status": "Healthy", "Green Percentage": 80, "Predicted Canopy": 1.6,
             "Plant Height": {"Current": 0.25, "Predicted": 0.27},
             "Longest Leaf": {"Current": 0.20, "Predicted": 0.22},
             "Yield": {"Current": 165, "Predicted": 180},
             "Leaves Count": {"Current": 10, "Predicted": 12}},
            {"Crop": 2, "Health Status": "Unhealthy", "Green Percentage": 45, "Predicted Canopy": 1.3,
             "Plant Height": {"Current": 0.21, "Predicted": 0.22},
             "Longest Leaf": {"Current": 0.17, "Predicted": 0.18},
             "Yield": {"Current": 140, "Predicted": 145},
             "Leaves Count": {"Current": 8, "Predicted": 9}},
            {"Crop": 3, "Health Status": "Healthy", "Green Percentage": 90, "Predicted Canopy": 1.7,
             "Plant Height": {"Current": 0.26, "Predicted": 0.28},
             "Longest Leaf": {"Current": 0.22, "Predicted": 0.24},
             "Yield": {"Current": 190, "Predicted": 200},
             "Leaves Count": {"Current": 12, "Predicted": 14}},
            {"Crop": 4, "Health Status": "Unhealthy", "Green Percentage": 35, "Predicted Canopy": 1.2,
             "Plant Height": {"Current": 0.18, "Predicted": 0.19},
             "Longest Leaf": {"Current": 0.16, "Predicted": 0.17},
             "Yield": {"Current": 110, "Predicted": 115},
             "Leaves Count": {"Current": 7, "Predicted": 8}},
            {"Crop": 5, "Health Status": "Healthy", "Green Percentage": 70, "Predicted Canopy": 1.5,
             "Plant Height": {"Current": 0.24, "Predicted": 0.26},
             "Longest Leaf": {"Current": 0.19, "Predicted": 0.21},
             "Yield": {"Current": 170, "Predicted": 185},
             "Leaves Count": {"Current": 11, "Predicted": 13}},
            {"Crop": 6, "Health Status": "Unhealthy", "Green Percentage": 25, "Predicted Canopy": 1.1,
             "Plant Height": {"Current": 0.20, "Predicted": 0.21},
             "Longest Leaf": {"Current": 0.15, "Predicted": 0.16},
             "Yield": {"Current": 95, "Predicted": 100},
             "Leaves Count": {"Current": 6, "Predicted": 7}},
            {"Crop": 7, "Health Status": "Healthy", "Green Percentage": 85, "Predicted Canopy": 1.8,
             "Plant Height": {"Current": 0.27, "Predicted": 0.29},
             "Longest Leaf": {"Current": 0.23, "Predicted": 0.25},
             "Yield": {"Current": 200, "Predicted": 215},
             "Leaves Count": {"Current": 13, "Predicted": 15}},
            {"Crop": 8, "Health Status": "Unhealthy", "Green Percentage": 50, "Predicted Canopy": 1.4,
             "Plant Height": {"Current": 0.22, "Predicted": 0.23},
             "Longest Leaf": {"Current": 0.18, "Predicted": 0.19},
             "Yield": {"Current": 150, "Predicted": 155},
             "Leaves Count": {"Current": 9, "Predicted": 10}},
            {"Crop": 9, "Health Status": "Healthy", "Green Percentage": 95, "Predicted Canopy": 1.9,
             "Plant Height": {"Current": 0.28, "Predicted": 0.30},
             "Longest Leaf": {"Current": 0.24, "Predicted": 0.26},
             "Yield": {"Current": 210, "Predicted": 225},
             "Leaves Count": {"Current": 14, "Predicted": 16}},
            {"Crop": 10, "Health Status": "Unhealthy", "Green Percentage": 40, "Predicted Canopy": 1.2,
             "Plant Height": {"Current": 0.20, "Predicted": 0.21},
             "Longest Leaf": {"Current": 0.16, "Predicted": 0.17},
             "Yield": {"Current": 120, "Predicted": 125},
             "Leaves Count": {"Current": 7, "Predicted": 8}}
        ]

        selected_crop_info = st.selectbox("Select a Crop to Highlight", range(1, 11))

        col1, col2 = st.columns(2)
        with col1:
            # Select a crop to highlight
            st.subheader('Current')
            # Plot the 3D crops
            plot_3d_crops(crop_health, selected_crop_info)

        with col2:
            # Display crop details for the selected crop
            st.subheader('Predicted')
            # Select a crop to highlight
            st.markdown("""
                <div style="font-size:18px; color: red; border: 2px solid #FF5733; padding: 15px; border-radius: 10px; background-color: #f9f9f9;">
                    <p style="text-align: center;"><strong>Upgrade to Premium Edition for full functionality.</strong></p>
                    <p style="text-align: center;">Contact <a href="mailto:nurfadhlina@upm.edu.my">nurfadhlina@upm.edu.my</a> for further details.</p>
                </div>
            """, unsafe_allow_html=True)

            # Plot the 3D crops


        if selected_crop_info:
            selected_crop_data = next((crop for crop in crop_health if crop["Crop"] == selected_crop_info), None)

            if selected_crop_data:
                # HTML content for displaying crop details
                # HTML content for displaying crop details
                html_content = f"""
                <div style="border: 2px solid #4CAF50; border-radius: 15px; padding: 20px; background-color: #f9f9f9;">
                    <h4 style="color: #4CAF50; text-align: center;">Crop Information</h4>
                    <p><b>Health Status:</b> {selected_crop_data['Health Status']}</p>
                    <div style="font-size:18px; color: red; border: 2px solid #FF5733; padding: 15px; border-radius: 10px; background-color: #f9f9f9;">
            <p style="text-align: center;"><strong>Upgrade to Premium Edition for full functionality.</strong></p>
            <p style="text-align: center;">Get access to the following premium features:</p>
            <ul style="font-size:16px;">
                <li><strong>Estimated Yield for the Area</strong>: Get precise yield predictions for different areas of your crop system.</li>
                <li><strong>Percentage of Target Achieved</strong>: Measure and track your crop's performance against target goals.</li>
                <li><strong>Green Percentage</strong>: Get detailed analysis on the green percentage in your crops for better management.</li>
                <li><strong>Predicted Canopy Size</strong>: Estimate the growth and canopy size of crops for better planning.</li>
                <li><strong>Current and Predicted Values</strong>: Compare current crop status with predicted values for accurate insights.</li>
            </ul>
            <p style="text-align: center;">Contact <a href="mailto:nurfadhlina@upm.edu.my">nurfadhlina@upm.edu.my</a> for further details.</p>
        </div><h5 style="color: #4CAF50; margin-top: 10px; text-align: center;">Explanation</h5>
                    <p style="text-align: justify;">
                        This section provides detailed insights into the health and growth metrics of the selected crop. 
                        The <b>Health Status</b> and <b>Green Percentage</b> indicate the overall vitality of the crop. 
                        The <b>Predicted Canopy Size</b> suggests the expected size of the crop canopy in square meters.
                        The table highlights the current and predicted values for key attributes such as plant height, 
                        longest leaf size, yield, and leaf count. By comparing these values, you can track the crop's growth trends 
                        and make informed decisions for better yield optimization.
                    </p>
                </div>
                """

                # Render the HTML content in Streamlit
                st.markdown(html_content, unsafe_allow_html=True)
        import streamlit as st
        import matplotlib.pyplot as plt

        # Example data for a selected crop (replace with actual crop data)
        crop_weekly_data = {
            "Week": [2, 3, 4],
            "Plant Height": [0.22, 0.25, 0.27],  # Height in meters
            "Longest Leaf": [0.18, 0.20, 0.22],  # Length in meters
            "Leaves Count": [8, 10, 12]  # Count of leaves
        }

        # Function to plot line graphs
        import matplotlib.pyplot as plt
        import streamlit as st

        def plot_line_graph(x, y, title, xlabel, ylabel, line_color="blue", marker_color="red"):
            """
            Plots a line graph with enhanced styling and better visualization.

            Parameters:
                x (list): Data for the x-axis (e.g., weeks).
                y (list): Data for the y-axis (e.g., plant height).
                title (str): Title of the graph.
                xlabel (str): Label for the x-axis.
                ylabel (str): Label for the y-axis.
                line_color (str): Color of the line. Default is "blue".
                marker_color (str): Color of the markers. Default is "red".
            """
            plt.figure(figsize=(8, 5))

            # Plot the line graph
            plt.plot(x, y, marker='o', linestyle='-', linewidth=2, color=line_color, markerfacecolor=marker_color,
                     markersize=8)

            # Title and axis labels
            plt.title(title, fontsize=14, fontweight='bold', color="#4CAF50", pad=10)
            plt.xlabel(xlabel, fontsize=12, fontweight='bold', color="#333")
            plt.ylabel(ylabel, fontsize=12, fontweight='bold', color="#333")

            # Add gridlines for better readability
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

            # Customize the x and y ticks for better visibility
            plt.xticks(fontsize=10, fontweight='bold', color="#555")
            plt.yticks(fontsize=10, fontweight='bold', color="#555")

            # Adjust layout for a clean appearance
            plt.tight_layout()

            # Display the graph in Streamlit
            st.pyplot(plt)

        # "More Info" button
        if st.button("More Info"):
            st.markdown("""
                    <div style="font-size:18px; color: red; border: 2px solid #FF5733; padding: 15px; border-radius: 10px; background-color: #f9f9f9;">
                        <p style="text-align: center;"><strong>Upgrade to Premium Edition for full functionality.</strong></p>                    
                        <p style="text-align: center;">Contact <a href="mailto:nurfadhlina@upm.edu.my">nurfadhlina@upm.edu.my</a> for further details.</p>
                    </div>
                """, unsafe_allow_html=True)


    def plot_2d_areaArea2():
        import plotly.graph_objects as go
        import streamlit as st

        # Example crop health data with updated values
        crop_health = [
            {"Crop": 1, "Health Status": "Healthy", "Green Percentage": 80, "Predicted Canopy": 1.6,
             "Plant Height": {"Current": 0.25, "Predicted": 0.27},
             "Longest Leaf": {"Current": 0.20, "Predicted": 0.22},
             "Yield": {"Current": 165, "Predicted": 180},
             "Leaves Count": {"Current": 10, "Predicted": 12}},
            {"Crop": 2, "Health Status": "Unhealthy", "Green Percentage": 45, "Predicted Canopy": 1.3,
             "Plant Height": {"Current": 0.21, "Predicted": 0.22},
             "Longest Leaf": {"Current": 0.17, "Predicted": 0.18},
             "Yield": {"Current": 140, "Predicted": 145},
             "Leaves Count": {"Current": 8, "Predicted": 9}},
            {"Crop": 3, "Health Status": "Healthy", "Green Percentage": 90, "Predicted Canopy": 1.7,
             "Plant Height": {"Current": 0.26, "Predicted": 0.28},
             "Longest Leaf": {"Current": 0.22, "Predicted": 0.24},
             "Yield": {"Current": 190, "Predicted": 200},
             "Leaves Count": {"Current": 12, "Predicted": 14}},
            {"Crop": 4, "Health Status": "Unhealthy", "Green Percentage": 35, "Predicted Canopy": 1.2,
             "Plant Height": {"Current": 0.18, "Predicted": 0.19},
             "Longest Leaf": {"Current": 0.16, "Predicted": 0.17},
             "Yield": {"Current": 110, "Predicted": 115},
             "Leaves Count": {"Current": 7, "Predicted": 8}},
            {"Crop": 5, "Health Status": "Healthy", "Green Percentage": 70, "Predicted Canopy": 1.5,
             "Plant Height": {"Current": 0.24, "Predicted": 0.26},
             "Longest Leaf": {"Current": 0.19, "Predicted": 0.21},
             "Yield": {"Current": 170, "Predicted": 185},
             "Leaves Count": {"Current": 11, "Predicted": 13}},
            {"Crop": 6, "Health Status": "Unhealthy", "Green Percentage": 25, "Predicted Canopy": 1.1,
             "Plant Height": {"Current": 0.20, "Predicted": 0.21},
             "Longest Leaf": {"Current": 0.15, "Predicted": 0.16},
             "Yield": {"Current": 95, "Predicted": 100},
             "Leaves Count": {"Current": 6, "Predicted": 7}},
            {"Crop": 7, "Health Status": "Healthy", "Green Percentage": 85, "Predicted Canopy": 1.8,
             "Plant Height": {"Current": 0.27, "Predicted": 0.29},
             "Longest Leaf": {"Current": 0.23, "Predicted": 0.25},
             "Yield": {"Current": 200, "Predicted": 215},
             "Leaves Count": {"Current": 13, "Predicted": 15}},
            {"Crop": 8, "Health Status": "Unhealthy", "Green Percentage": 50, "Predicted Canopy": 1.4,
             "Plant Height": {"Current": 0.22, "Predicted": 0.23},
             "Longest Leaf": {"Current": 0.18, "Predicted": 0.19},
             "Yield": {"Current": 150, "Predicted": 155},
             "Leaves Count": {"Current": 9, "Predicted": 10}},
            {"Crop": 9, "Health Status": "Healthy", "Green Percentage": 95, "Predicted Canopy": 1.9,
             "Plant Height": {"Current": 0.28, "Predicted": 0.30},
             "Longest Leaf": {"Current": 0.24, "Predicted": 0.26},
             "Yield": {"Current": 210, "Predicted": 225},
             "Leaves Count": {"Current": 14, "Predicted": 16}},
            {"Crop": 10, "Health Status": "Unhealthy", "Green Percentage": 40, "Predicted Canopy": 1.2,
             "Plant Height": {"Current": 0.20, "Predicted": 0.21},
             "Longest Leaf": {"Current": 0.16, "Predicted": 0.17},
             "Yield": {"Current": 120, "Predicted": 125},
             "Leaves Count": {"Current": 7, "Predicted": 8}}
        ]

        def plot_3d_crops(crop_health, selected_crop):
            """
            Visualizes crop health using 3D lines with spheres on top.

            Parameters:
                crop_health (list): List of dictionaries with crop health data.
                selected_crop (int): Index of the selected crop (1-based).
            """
            # Ensure crop_health contains 10 items (add placeholders if necessary)
            while len(crop_health) < 10:
                crop_health.append({
                    "Crop": len(crop_health) + 1,
                    "Health Status": "Not Detected",
                    "Green Pixels": 0,
                    "Total Pixels": 0,
                    "Green Percentage": 0
                })

            # Initialize 3D plot
            fig = go.Figure()

            # Define 3D positions for the crops (spread on the X-Y plane for 10 crops)
            x_positions = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
            y_positions = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
            z_positions = [0] * 10  # Base height of lines (z = 0)

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
                showlegend=False
            )

            # Display the 3D plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)


        col1, col2 = st.columns(2)
        with col1:
            # Select a crop to highlight
            selected_crop_info = st.selectbox("Select a Crop to Highlight", range(1, 11))

            # Plot the 3D crops
            plot_3d_crops(crop_health, selected_crop_info)

        with col2:
            # Display crop details for the selected crop
            if selected_crop_info:
                selected_crop_data = next((crop for crop in crop_health if crop["Crop"] == selected_crop_info), None)

                if selected_crop_data:
                    # HTML content for displaying crop details
                    # HTML content for displaying crop details
                    html_content = f"""
                    <div style="border: 2px solid #4CAF50; border-radius: 15px; padding: 20px; background-color: #f9f9f9;">
                        <h4 style="color: #4CAF50; text-align: center;">Crop Information</h4>
                        <p><b>Health Status:</b> {selected_crop_data['Health Status']}</p>
                        <p><b>Green Percentage:</b> {selected_crop_data['Green Percentage']}%</p>
                        <p><b>Predicted Canopy Size:</b> {selected_crop_data['Predicted Canopy']} m²</p>
                        <hr style="border-top: 1px solid #4CAF50;">
                        <h5 style="color: #4CAF50; margin-top: 10px; text-align: center;">Current and Predicted Values</h5>
                        <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                            <tr>
                                <th style="border: 1px solid #ddd; padding: 8px; background-color: #4CAF50; color: white;">Attribute</th>
                                <th style="border: 1px solid #ddd; padding: 8px; background-color: #4CAF50; color: white;">Current Value</th>
                                <th style="border: 1px solid #ddd; padding: 8px; background-color: #4CAF50; color: white;">Predicted Value</th>
                            </tr>
                            <tr>
                                <td style="border: 1px solid #ddd; padding: 8px;">Plant Height</td>
                                <td style="border: 1px solid #ddd; padding: 8px;">{selected_crop_data['Plant Height']['Current']} m</td>
                                <td style="border: 1px solid #ddd; padding: 8px;">{selected_crop_data['Plant Height']['Predicted']} m</td>
                            </tr>
                            <tr>
                                <td style="border: 1px solid #ddd; padding: 8px;">Longest Leaf</td>
                                <td style="border: 1px solid #ddd; padding: 8px;">{selected_crop_data['Longest Leaf']['Current']} m</td>
                                <td style="border: 1px solid #ddd; padding: 8px;">{selected_crop_data['Longest Leaf']['Predicted']} m</td>
                            </tr>
                            <tr>
                                <td style="border: 1px solid #ddd; padding: 8px;">Yield</td>
                                <td style="border: 1px solid #ddd; padding: 8px;">{selected_crop_data['Yield']['Current']} units</td>
                                <td style="border: 1px solid #ddd; padding: 8px;">{selected_crop_data['Yield']['Predicted']} units</td>
                            </tr>
                            <tr>
                                <td style="border: 1px solid #ddd; padding: 8px;">Leaves Count</td>
                                <td style="border: 1px solid #ddd; padding: 8px;">{selected_crop_data['Leaves Count']['Current']}</td>
                                <td style="border: 1px solid #ddd; padding: 8px;">{selected_crop_data['Leaves Count']['Predicted']}</td>
                            </tr>
                        </table>
                        <hr style="border-top: 1px solid #4CAF50;">
                        <h5 style="color: #4CAF50; margin-top: 10px; text-align: center;">Explanation</h5>
                        <p style="text-align: justify;">
                            This section provides detailed insights into the health and growth metrics of the selected crop. 
                            The <b>Health Status</b> and <b>Green Percentage</b> indicate the overall vitality of the crop. 
                            The <b>Predicted Canopy Size</b> suggests the expected size of the crop canopy in square meters.
                            The table highlights the current and predicted values for key attributes such as plant height, 
                            longest leaf size, yield, and leaf count. By comparing these values, you can track the crop's growth trends 
                            and make informed decisions for better yield optimization.
                        </p>
                    </div>
                    """

                    # Render the HTML content in Streamlit
                    st.markdown(html_content, unsafe_allow_html=True)


    areas = ["Area 1", "Area 2"]

    # Create a dropdown in Streamlit
    selected_area = st.selectbox("Select the Area:", options=areas)
    if selected_area == "Area 1":
        st.subheader("Crop Health Analyze")
        plot_2d_areaArea1()
        recommendations = [
            {
                "Area": f"Area {i + 1}",
                "Weekly Recommendation": {
                    "Week 2": f"Solution A: {250 + i * 10} mml + Solution B: {350 - i * 5} mml to adjust pH and EC.",
                    "Action": "Crop 5 should be cut and replaced." if i % 3 == 0 else "No crops need replacing.",
                    "Pests": "Pests not detected." if i % 2 == 0 else "Pests detected! Apply pest control measures.",
                },
                "Future Recommendations": {
                    "Week 3": {
                        "Solution A": f"{200 + i * 5} mml",
                        "Solution B": f"{300 - i * 10} mml",
                        "Instructions": "Adjust Solution B for stable EC levels.",
                    },
                    "Week 4": {
                        "Solution A": f"{150 + i * 2} mml",
                        "Solution B": f"{250 - i * 7} mml",
                        "Instructions": "Maintain pH using Solution A only.",
                    },
                },
            }
            for i in range(19)
        ]

        # Dropdown for selecting an area
        selected_area_name = selected_area

        # Find the selected area data
        selected_area = next(area for area in recommendations if area["Area"] == selected_area_name)

        # Display selected area recommendations
        st.markdown(
            f"""
                    <div class="recommendation-box">
                        <h4 style="color: #4CAF50; text-align: center;">Recommendations {selected_area['Area']}</h4>
                        <div style="font-size:18px; color: red; border: 2px solid #FF5733; padding: 15px; border-radius: 10px; background-color: #f9f9f9;">
                                                <p style="text-align: center;"><strong>Upgrade to Premium Edition for full functionality.</strong></p>
                                                <p style="text-align: center;">Contact <a href="mailto:nurfadhlina@upm.edu.my">nurfadhlina@upm.edu.my</a> for further details.</p>                                                
                    </div>
                    """,
            unsafe_allow_html=True,
        )

        # Toggle for More Info
        if st.button("📖 Show More Info"):
            st.markdown(
                f"""
                                            <div class="recommendation-box">
                                                <h4 style="color: #4CAF50; text-align: center;">Recommendations Next Weeks</h4>
                                                <div style="font-size:18px; color: red; border: 2px solid #FF5733; padding: 15px; border-radius: 10px; background-color: #f9f9f9;">
                                                <p style="text-align: center;"><strong>Upgrade to Premium Edition for full functionality.</strong></p>
                                                <p style="text-align: center;">Contact <a href="mailto:nurfadhlina@upm.edu.my">nurfadhlina@upm.edu.my</a> for further details.</p>
                                            </div>
                                            """,
                unsafe_allow_html=True,
            )

        # CSS for Styling
        st.markdown(
            """
            <style>
            .recommendation-box {
                border: 3px solid #4CAF50;
                padding: 20px;
                margin: 10px 0;
                border-radius: 10px;
                background-color: #f9f9f9;
                animation: fadeIn 1.5s ease-in-out;
            }
            .more-info {
                margin-top: 10px;
                padding: 10px;
                background-color: #e0f7fa;
                border: 1px solid #4CAF50;
                border-radius: 5px;
                font-size: 1em;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 1em;
            }
            button:hover {
                background-color: #45a049;
            }
            @keyframes fadeIn {
                from {opacity: 0;}
                to {opacity: 1;}
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Display the 2D area visualization with the selected crop highlighted
    # Additional logic or data visualization for Area 1
    elif selected_area == "Area 2":
        plot_2d_areaArea2()
        recommendations = [
            {
                "Area": f"Area {i + 1}",
                "Weekly Recommendation": {
                    "Week 2": f"Solution A: {250 + i * 10} mml + Solution B: {350 - i * 5} mml to adjust pH and EC.",
                    "Action": "Crop 5 should be cut and replaced." if i % 3 == 0 else "No crops need replacing.",
                    "Pests": "Pests not detected." if i % 2 == 0 else "Pests detected! Apply pest control measures.",
                },
                "Future Recommendations": {
                    "Week 3": {
                        "Solution A": f"{200 + i * 5} mml",
                        "Solution B": f"{300 - i * 10} mml",
                        "Instructions": "Adjust Solution B for stable EC levels.",
                    },
                    "Week 4": {
                        "Solution A": f"{150 + i * 2} mml",
                        "Solution B": f"{250 - i * 7} mml",
                        "Instructions": "Maintain pH using Solution A only.",
                    },
                },
            }
            for i in range(19)
        ]

        # Dropdown for selecting an area
        selected_area_name = area_selected

        # Find the selected area data
        selected_area = next(area for area in recommendations if area["Area"] == selected_area_name)

        # Display selected area recommendations
        st.markdown(
            f"""
                    <div class="recommendation-box">
                        <h4 style="color: #4CAF50; text-align: center;">Recommendations {selected_area['Area']}</h4>
                        <div style="font-size:18px; color: red; border: 2px solid #FF5733; padding: 15px; border-radius: 10px; background-color: #f9f9f9;">
                        <p style="text-align: center;"><strong>Upgrade to Premium Edition for full functionality.</strong></p>
                        <p style="text-align: center;">Contact <a href="mailto:nurfadhlina@upm.edu.my">nurfadhlina@upm.edu.my</a> for further details.</p>
                    </div>
                    """,
            unsafe_allow_html=True,
        )

        # Toggle for More Info
        if st.button("📖 Show More Info"):
            st.markdown(
                f"""
                                <div class="recommendation-box">
                                    <h4 style="color: #4CAF50; text-align: center;">Recommendations {selected_area['Area']}</h4>
                                    <div style="font-size:18px; color: red; border: 2px solid #FF5733; padding: 15px; border-radius: 10px; background-color: #f9f9f9;">
                                    <p style="text-align: center;"><strong>Upgrade to Premium Edition for full functionality.</strong></p>
                                    <p style="text-align: center;">Contact <a href="mailto:nurfadhlina@upm.edu.my">nurfadhlina@upm.edu.my</a> for further details.</p>
                                </div>
                                """,
                unsafe_allow_html=True,
            )

        # Display the 2D area visualization with the selected crop highlighted


