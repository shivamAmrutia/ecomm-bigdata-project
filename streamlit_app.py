import streamlit as st
from PIL import Image
from src.predict import predict_purchase
import json


# app
st.set_page_config(layout="wide", page_title="E-Commerce Funnel Dashboard")

st.title("🛒 E-Commerce Behavior Dashboard")

# --- Tabs ---
tab1, tab2 = st.tabs(["📈 Visualizations", "🔍 Simulate & Predict"])

# --- Tab 1: Visualizations ---
with tab1:
    st.subheader("User Funnel Analysis & Behavior")

    col1, col2 = st.columns(2)
    with col1:
        st.image("./output/figures/conversion_funnel.png", caption="Conversion Funnel", use_column_width=True)
        st.image("./output/figures/dropoff_rate.png", caption="Drop-Off Rate", use_column_width=True)
    with col2:
        st.image("./output/figures/purchase_probability.png", caption="Predicted Probabilities", use_column_width=True)
        st.image("./output/figures/session_duration_vs_purchase.png", caption="Session Duration vs Purchase", use_column_width=True)
        st.image("./output/figures/price_vs_purchase.png", caption="Price Sensitivity", use_column_width=True)

# --- Tab 2: Predict from Simulated Input ---

# getting best model and params
# with open("./output/best_params.json") as f:
#     best_model_and_params = json.load(f)
# model = best_model_and_params.get("model")
# params = best_model_and_params.get("params")

# with tab2:
#     st.subheader("Simulate a User Session")

#     num_views = st.slider("Number of Product Views", 0, 20, 3)
#     num_cart_adds = st.slider("Number of Cart Adds", 0, 10, 1)
#     num_purchases = st.slider("Number of Past Purchases in Session", 0, 3, 0)
#     avg_price = st.number_input("Average Product Price Viewed", 10.0, 1000.0, 150.0)
    
#     features = [num_views, num_cart_adds, num_purchases, avg_price]

#     if st.button("Predict Purchase Likelihood"):
#         prob = predict_purchase(features, model, params)
#         st.success(f"🧠 Predicted Probability of Purchase: **{prob:.2%}**")
