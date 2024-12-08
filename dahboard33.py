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
            st.write(f"**Plant Height:** {selected_crop_data['Plant Height']} m")
            st.write(f"**Longest Leaf:** {selected_crop_data['Longest Leaf']} m")
            st.write(f"**Yield in Harvest:** {selected_crop_data['Yield']} g")
