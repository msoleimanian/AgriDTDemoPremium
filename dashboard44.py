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
        "Poor": "#FF0000",     # Red
        "Weak": "#FFA500",     # Orange
        "Normal": "#FFFF00",   # Yellow
        "Good": "#4CAF50",     # Green
        "Excellent": "#008000" # Dark Green
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
        circular_progress_bar(desired_yield_percentage, "Desired Yield", "1.4 KG", "#FFA500"),  # Orange for desired yield
        unsafe_allow_html=True,
    )

# Line progress bar for Pak Choy Growth (Week 1)
st.markdown(
    line_progress_bar(pak_choy_progress, f"Pak Choy Growth (Week {pak_choy_week})", f"{pak_choy_week} Week", get_health_category(pak_choy_progress), 100),
    unsafe_allow_html=True,
)


