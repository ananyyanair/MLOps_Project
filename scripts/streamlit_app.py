import streamlit as st
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.predict import predict
from loguru import logger
logger.add("logs/app.log", rotation="500 KB")

st.title("📱 Mobile Price Predictor")
st.markdown("Enter the mobile features below to predict its price:")

# Create two columns for better layout
col1, col2 = st.columns(2)

# Default values
default_values = {
    'battery': 4000,
    'ram': 4096,
    'display': 6.5,
    'camera': 48,
    'inbuilt_memory': 128,
    'fast_charging': 33,
    'external_memory': True,
    'screen_resolution': "1080x2400",
    'sim_features': ["Dual Sim", "4G"],
    'android_version': "13.0",
    'company': "Samsung",
    'processor': "Snapdragon",
    'processor_name': "Snapdragon 888"
}

with col1:
    st.subheader("Hardware Specifications")
    battery = st.number_input("Battery (mAh)", min_value=500, max_value=7000, 
                             value=default_values['battery'], step=100)
    ram = st.number_input("RAM (MB)", min_value=512, max_value=16384, 
                         value=default_values['ram'], step=512)
    display = st.number_input("Display Size (inches)", min_value=4.0, max_value=7.5, 
                             value=default_values['display'], step=0.1)
    camera = st.number_input("Total Camera MP", min_value=2, max_value=200, 
                            value=default_values['camera'], step=1)
    inbuilt_memory = st.number_input("Inbuilt Storage (GB)", min_value=8, max_value=512, 
                                    value=default_values['inbuilt_memory'], step=8)
    fast_charging = st.number_input("Fast Charging (Watts)", min_value=0, max_value=150, 
                                   value=default_values['fast_charging'], step=5)

with col2:
    st.subheader("Other Features")
    external_memory = st.radio("External Memory Supported?", ["Yes", "No"], 
                              index=0 if default_values['external_memory'] else 1) == "Yes"
    
    screen_resolution_options = ["720x1600", "1080x2400", "1440x3200", "1440x3120", "1170x2532"]
    if default_values['screen_resolution'] not in screen_resolution_options:
        screen_resolution_options.append(default_values['screen_resolution'])
    
    screen_resolution = st.selectbox("Screen Resolution", 
                                   screen_resolution_options,
                                   index=screen_resolution_options.index(default_values['screen_resolution']))
    
    sim_features = st.multiselect("SIM & Network Features", 
                                ["Dual Sim", "3G", "4G", "5G", "VoLTE"], 
                                default=default_values['sim_features'])
    android_version = st.selectbox("Android Version", 
                                 ["9.0", "10.0", "11.0", "12.0", "13.0", "14.0", "15.0"], 
                                 index=["9.0", "10.0", "11.0", "12.0", "13.0", "14.0", "15.0"].index(default_values['android_version']))
    company = st.selectbox("Company", 
                          ["Samsung", "Apple", "Xiaomi", "OnePlus", "Vivo", "Oppo", "Realme"],
                          index=["Samsung", "Apple", "Xiaomi", "OnePlus", "Vivo", "Oppo", "Realme"].index(default_values['company']))
    processor = st.selectbox("Processor Brand", 
                           ["Snapdragon", "MediaTek", "Exynos", "Apple"],
                           index=["Snapdragon", "MediaTek", "Exynos", "Apple"].index(default_values['processor']))
    
    # Processor names based on brand
    processor_names = {
        "Snapdragon": ["Snapdragon 8 Gen 1", "Snapdragon 888", "Snapdragon 870", "Snapdragon 855"],
        "MediaTek": ["Dimensity 1200", "Dimensity 900", "Helio G95", "Helio P95"],
        "Exynos": ["Exynos 2100", "Exynos 990", "Exynos 9825"],
        "Apple": ["A15 Bionic", "A14 Bionic", "A13 Bionic"]
    }
    
    available_processors = processor_names.get(processor, ["Generic Processor"])
    if default_values['processor_name'] in available_processors:
        default_index = available_processors.index(default_values['processor_name'])
    else:
        default_index = 0
    
    processor_name = st.selectbox("Processor Name", available_processors, index=default_index)

# Create input dictionary
input_dict = {
    "Battery": battery,
    "Ram": ram,
    "Display": display,
    "Camera": camera,
    "External_Memory": external_memory,
    "Inbuilt_memory": inbuilt_memory,
    "fast_charging": fast_charging,
    "Screen_resolution": screen_resolution,
    "No_of_sim": sim_features,
    "Android_version": android_version,
    "company": company,
    "Processor": processor,
    "Processor_name": processor_name
}

# Prediction button
if st.button("🔮 Predict Price", type="primary"):
    try:
        with st.spinner("Predicting price..."):
            prediction = predict(input_dict)
            
            # Display result with styling
            st.success("✅ Prediction Completed!")
            st.metric(
                label="Predicted Mobile Price", 
                value=f"₹{prediction[0]:,.2f}",
                delta=None
            )
            
            # Add price category context
            if prediction[0] < 15000:
                st.info("💡 This appears to be a budget-friendly phone")
            elif prediction[0] < 30000:
                st.info("💡 This appears to be a mid-range phone")
            elif prediction[0] < 60000:
                st.info("💡 This appears to be a premium phone")
            else:
                st.info("💡 This appears to be a flagship/ultra-premium phone")
                
            logger.info(f"Successful prediction: ₹{prediction[0]:,.2f}")
            
    except Exception as e:
        st.error("❌ Prediction failed. Please check the logs for details.")
        st.error(f"Error details: {str(e)}")
        logger.error(f"Streamlit prediction error: {e}")

# Minimal help information
with st.expander("ℹ️ Help"):
    st.markdown("""
    **How to use:**
    1. Adjust hardware specifications in left column
    2. Select other features in right column
    3. Click 'Predict Price' for estimated price
    
    **Notes:**
    - RAM values in MB (4096 MB = 4GB)
    - Camera MP represents total megapixels
    - iPhone: Use closest Android version equivalent
    """)