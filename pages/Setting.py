import streamlit as st

# Sidebar Dropdown for Settings
st.sidebar.title("Settings")
settings_option = st.sidebar.selectbox(
    "Choose an Action:",  # Dropdown title
    ["Select an option", "Add Farm & Product"],  # Options
)

# Logic for Settings Page
if settings_option == "Add Farm & Product":
    st.title("Add Farm & Product")

    # Section for Adding Farm
    st.subheader("Add Farm")
    farm_name = st.text_input("Enter Farm Name:")
    farm_location = st.text_input("Enter Farm Location:")
    if st.button("Save Farm"):
        st.success(f"Farm '{farm_name}' at location '{farm_location}' added successfully!")

    st.markdown("---")  # Divider

    # Section for Adding Product
    st.subheader("Add Product")
    product_name = st.text_input("Enter Product Name:")
    product_category = st.text_input("Enter Product Category:")
    if st.button("Save Product"):
        st.success(f"Product '{product_name}' in category '{product_category}' added successfully!")

else:
    st.title("Settings")
    st.write("Choose an action from the dropdown in the sidebar.")
