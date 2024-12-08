import streamlit as st



html_content = f"""
                                <div style="border: 2px solid #333333; padding:10px; border-radius:5px;">
                                                    <h3 style="color:#333333;">What does the system do?</h3>
                                                    <p>Welcome to the Home page of our innovative system, designed to revolutionize the way you manage and analyze your agricultural operations. Our system comprises Three main menus, each offering a distinct view into the world of smart farming: Historical Analyses and Digital Twin.</p>
                                                </div>
                                            """
st.markdown(html_content, unsafe_allow_html=True)

st.write('')
col1, col2 = st.columns(2)
with col1:
    html_content = f"""
                                        <div style="border: 2px solid #333333; padding:10px; border-radius:5px;">
                                                            <h3 style="color:#333333;">Historical Analyses</h3>
                                                            <p>Unearth the insights hidden within your past data and discover the optimal conditions for plant growth and overall farm performance. This menu allows you to delve into the rich history of your agricultural endeavors, comparing different plantings and generations to identify patterns and trends.</p>
                                                        </div>
                                                    """
    st.markdown(html_content, unsafe_allow_html=True)

st.write('')
with col2:
    st.image("DataAnalyze2.jpeg", width=320)

col1, col2 = st.columns(2)
with col2:
    html_content = f"""
                                                <div style="border: 2px solid #333333; padding:10px; border-radius:5px;">
                                                                    <h3 style="color:#333333;">Digital Twin</h3>
                                                                    <p>Experience the cutting-edge technology of Digital Twin, where the virtual meets the real-time. Our system provides immersive 3D models of your greenhouse, offering a detailed and interactive view of each plant's current status.</p>
                                                                </div>
                                                            """
    st.markdown(html_content, unsafe_allow_html=True)

with col1:
    st.image("DigitalTwin.jpeg", width=350)
st.write('')

col1, col2, col3 = st.columns(3)

if col2.button('More Info'):
    html_content = f"""
                                            <div style="border: 2px solid #333333; padding:10px; border-radius:5px;">
                                                                <h3 style="color:#333333;">Historical Analyses</h3>
        1. Comparative Analysis:
        Explore the performance of different plantings and generations side by side. Our system provides detailed comparisons, highlighting which plants thrived under specific conditions. This feature enables you to make informed decisions based on historical success rates.

    2. Insights and Trends:
    Dive deep into the trends that shaped your farm's history. Identify the factors contributing to the best performance of your plants and understand how they relate to specific generations. Uncover valuable insights that can guide your future planting strategies.

    3. Generation Exploration:
    Navigate through the various generations of your crops. Examine the conditions, challenges, and successes of each generation. This feature empowers you to learn from the past and make data-driven decisions to optimize future generations.


                                                        """
    st.markdown(html_content, unsafe_allow_html=True)
    st.write('')
    html_content = f"""
                                            <div style="border: 2px solid #333333; padding:10px; border-radius:5px;">
                                                                <h3 style="color:#333333;">Digital Twin</h3>

1. Real-Time Monitoring:
Witness the live status of your crops through three-dimensional models of your greenhouse. Sensors strategically placed in your farm provide real-time updates on temperature, humidity, soil moisture, and more. Stay connected to your farm like never before.

2. Predictive Analysis:
Anticipate the future status of your farm with our predictive analysis. By analyzing current conditions and historical data, our system forecasts potential issues and suggests preemptive measures. Proactively manage your farm for optimal performance.

3. Recommendations:
Receive personalized recommendations based on the real-time data and predictive analysis. Our system guides you on necessary actions to enhance the health and productivity of your crops. From adjusting environmental conditions to suggesting specific interventions, we've got you covered.</p>
                                                            </div>
                                                        """
    st.markdown(html_content, unsafe_allow_html=True)