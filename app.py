import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os
import sklearn  


st.set_page_config(page_title="Slope Stability Optimizer (AI)", layout="wide", page_icon="⛰️")

# --- CSS Styling ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
        background-color: #4e73df;
        color: white;
    }
    .metric-card {
        background-color: #f8f9fc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4e73df;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .stable-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .unstable-badge {
        background-color: #f8d7da;
        color: #721c24;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("AI Slope Stability Optimization Engine")
st.markdown("""
**Author**: Muhammed Yeasin Arafat, B.Sc. in Civil Engineering, BUET, Bangladesh.
""")

REINFORCEMENT_MAP = {
    0: "Retaining Wall",
    1: "Soil Nailing",
    2: "Geosynthetics",
    3: "Drainage"
}


st.sidebar.header("⚙️ System Configuration")
st.sidebar.markdown("Specify the paths to your saved artifacts.")


model_path_input = st.sidebar.text_input("Model Path (.h5)", "model_hybrid.h5")
scaler_path_input = st.sidebar.text_input("Scaler Path (.pkl)", "scaler.pkl")



@st.cache_resource
def load_prediction_system(model_path, scaler_path):
    """
    Loads the trained Keras model and the Scikit-Learn scaler.
    """
    model = None
    scaler = None
    error = None
    
    if not os.path.exists(model_path):
        return None, None, f"Model file not found at: {model_path}"
    if not os.path.exists(scaler_path):
        return None, None, f"Scaler file not found at: {scaler_path}"
        
    try:
        model = load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
    except Exception as e:
        error = str(e)
        
    return model, scaler, error


model, scaler, load_error = load_prediction_system(model_path_input, scaler_path_input)


if model and scaler:
    st.sidebar.success("System Online: Models Loaded")
else:
    st.sidebar.warning("System Offline: Waiting for valid files")
    if load_error:
        st.sidebar.error(f"Error: {load_error}")
    st.info("Please ensure your `.h5` model and `.pkl` scaler are in the directory and paths are correct.")
    st.stop()


st.markdown("### 1. Input Slope Parameters")

with st.form("analysis_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        unit_weight = st.number_input("Unit Weight (kN/m³)", 15.0, 25.0, 19.0)
        slope_angle = st.number_input("Slope Angle (°)", 10.0, 60.0, 45.0)
        
    with col2:
        cohesion = st.number_input("Cohesion (kPa)", 5.0, 50.0, 15.0)
        slope_height = st.number_input("Slope Height (m)", 5.0, 50.0, 20.0)
        
    with col3:
        friction = st.number_input("Friction Angle (°)", 20.0, 45.0, 30.0)
        pore_pressure = st.number_input("Pore Pressure Ratio (ru)", 0.0, 1.0, 0.5)

    submitted = st.form_submit_button("Run AI Analysis")

if submitted:
    st.markdown("---")
    st.markdown("### 2. Analysis Results")
    
    results = []
    
    try:        
        for code, name in REINFORCEMENT_MAP.items():
            

            scenario_input = np.array([[
                unit_weight, cohesion, friction, slope_angle, slope_height, pore_pressure, code
            ]])            
            
            scenario_scaled = scaler.transform(scenario_input)            
            
            if hasattr(model, 'input_shape') and len(model.input_shape) == 3:
                scenario_scaled = np.expand_dims(scenario_scaled, axis=1)            
            
            pred_fs = model.predict(scenario_scaled, verbose=0).flatten()[0]            
            
            status = "Stable" if pred_fs >= 1.0 else "Unstable"
            
            results.append({
                "Strategy": name,
                "Predicted FS": pred_fs,
                "Status": status
            })
                
        results_df = pd.DataFrame(results).sort_values(by="Predicted FS", ascending=False)        
        
        best_option = results_df.iloc[0]        
        
        col_res, col_chart = st.columns([2, 1])
        
        with col_res:
            st.subheader("Recommendation")
            
            if best_option['Predicted FS'] >= 1.0:
                st.success(f"**Optimal Solution:** Use **{best_option['Strategy']}**.")
                st.markdown(f"This strategy yields a Factor of Safety (FS) of **{best_option['Predicted FS']:.3f}**, ensuring stability.")
            else:
                st.error(f"**Critical Warning:** All reinforcement strategies failed to achieve FS >= 1.0.")
                st.markdown(f"The best available option is **{best_option['Strategy']}** (FS: {best_option['Predicted FS']:.3f}), but it is still UNSTABLE. Consider reducing the slope angle or height.")

            st.subheader("Strategy Comparison")            
           
            def style_df(val):
                color = '#d4edda' if val == 'Stable' else '#f8d7da'
                return f'background-color: {color}'

            st.dataframe(
                results_df.style.map(style_df, subset=['Status'])
                                .format({"Predicted FS": "{:.3f}"}),
                use_container_width=True,
                hide_index=True
            )
            
        with col_chart:
            st.subheader("FS Visualization")
            st.bar_chart(results_df.set_index("Strategy")["Predicted FS"])
            st.caption("Target Safety Factor > 1.0")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Check if your `.pkl` scaler matches the feature count (7 features expected) and the `.h5` model input shape.")