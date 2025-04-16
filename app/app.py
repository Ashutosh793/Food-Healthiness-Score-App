import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import requests
import os
import cv2
from pyzbar import pyzbar
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image
import logging

# ========== Logger Setup ==========
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ========== Load Models ==========
clf = joblib.load("C://Users//noobb//Downloads//Food healthiness project//models//classifier.pkl")
reg = joblib.load("C://Users//noobb//Downloads//Food healthiness project//models//regressor.pkl")

# ========== Load Dataset for Alternatives ==========
df = pd.read_csv("C://Users//noobb//Downloads//Food healthiness project//notebooks//engineered_food_data.csv")

# ========== Streamlit App Config ==========
st.set_page_config(page_title="🍎 Food Healthiness Scorer", layout="centered")
st.title("🍎 Food Healthiness Scoring System")

st.markdown("Enter nutrition facts and ingredient features to predict how healthy a food item is!")

# ========== Barcode Lookup via Sidebar ==========
st.sidebar.header("📷 Barcode Lookup")
barcode = st.sidebar.text_input("Enter barcode")

# ========== Upload Image for Barcode ==========
st.sidebar.subheader("📤 Upload Barcode Image")
uploaded = st.sidebar.file_uploader("Upload an image with a barcode", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded)
    decoded = pyzbar.decode(image)
    if decoded:
        detected_code = decoded[0].data.decode("utf-8")
        st.session_state.detected_barcode = detected_code
        st.sidebar.success(f"Detected Barcode from Image: {detected_code}")
        barcode = detected_code
    else:
        st.sidebar.error("No barcode detected in the uploaded image.")

# ========== Live Webcam Barcode Scanner ==========
st.sidebar.subheader("🎥 Live Barcode Scanner")
barcode_holder = st.sidebar.empty()

if 'detected_barcode' not in st.session_state:
    st.session_state.detected_barcode = ''

class BarcodeScanner(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        barcodes = pyzbar.decode(img)
        logger.debug(f"Detected barcodes: {barcodes}")
        for barcode in barcodes:
            barcode_data = barcode.data.decode("utf-8")
            st.session_state.detected_barcode = barcode_data
            barcode_holder.info(f"Detected Barcode: {barcode_data}")
        return img

webrtc_streamer(key="barcode", video_transformer_factory=BarcodeScanner)

# Use detected barcode from session_state if available
if st.session_state.detected_barcode:
    barcode = st.session_state.detected_barcode

if barcode:
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
    res = requests.get(url)

    if res.status_code == 200:
        data = res.json()
        if data.get("status") == 1:
            product = data["product"]
            product_name = product.get("product_name", "Unknown Product")
            product_image = product.get("image_url", "")

            st.sidebar.success(f"Found: {product_name}")
            if product_image:
                st.sidebar.image(product_image, caption=product_name, use_column_width=True)

            def get_nutrient(nutr_name):
                return product.get("nutriments", {}).get(nutr_name + "_100g", 0)

            ingredients_text = product.get("ingredients_text", "")
            additive_tags = product.get("additives_tags", [])
            ingredients_list = ingredients_text.split(",")

            def count_bad_ingredients(text):
                bad_ingredients = ['sugar', 'fructose', 'glucose', 'syrup', 'palm oil', 'maltodextrin']
                return sum([text.lower().count(word) for word in bad_ingredients])

            input_features = [
                get_nutrient("energy-kcal"),
                get_nutrient("fat"),
                get_nutrient("sugars"),
                get_nutrient("salt"),
                get_nutrient("fiber"),
                get_nutrient("proteins"),
                count_bad_ingredients(ingredients_text),
                len(additive_tags),
                len(ingredients_list)
            ]

            st.sidebar.write("📋 Features extracted:")
            for name, val in zip([
                'Energy', 'Fat', 'Sugars', 'Salt', 'Fiber', 'Proteins',
                'Bad Ingredient Count', 'Additives Count', 'Ingredients Length'
            ], input_features):
                st.sidebar.write(f"**{name}:** {val}")

            pred_label = clf.predict([input_features])[0]
            pred_score = reg.predict([input_features])[0]

            st.sidebar.markdown("---")
            st.sidebar.success(f"🏷️ Health Label: **{pred_label}**")
            st.sidebar.success(f"📉 NutriScore: **{pred_score:.2f}**")

        else:
            st.sidebar.error("❌ Product not found in OpenFoodFacts database.")
    else:
        st.sidebar.error("❌ Failed to fetch data from API.")

# ========== User Input Form ==========
st.subheader("📝 Manual Entry")
with st.form("health_form"):
    energy = st.number_input("Energy (kcal per 100g)", 0.0, 3000.0, 500.0)
    fat = st.number_input("Fat (g per 100g)", 0.0, 100.0, 10.0)
    sugar = st.number_input("Sugars (g per 100g)", 0.0, 100.0, 5.0)
    salt = st.number_input("Salt (g per 100g)", 0.0, 10.0, 0.5)
    fiber = st.number_input("Fiber (g per 100g)", 0.0, 20.0, 2.5)
    protein = st.number_input("Proteins (g per 100g)", 0.0, 50.0, 5.0)
    bad_ingredients = st.slider("# of Bad Ingredients", 0, 10, 1)
    additives = st.slider("# of Additives", 0, 10, 1)
    ingredients_len = st.slider("# of Ingredients (Length)", 0, 100, 10)
    submitted = st.form_submit_button("Predict Health Score")

# ========== Prediction from Form ==========
if submitted:
    input_data = np.array([[
        energy, fat, sugar, salt, fiber, protein,
        bad_ingredients, additives, ingredients_len
    ]])

    pred_label = clf.predict(input_data)[0]
    pred_score = reg.predict(input_data)[0]

    st.subheader("🧠 Predictions")
    st.write(f"**Predicted Health Label:** {pred_label}")
    st.write(f"**Predicted NutriScore:** {pred_score:.2f}")

    # ========== SHAP Explainability ==========
    st.subheader("🔍 Feature Importance")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(input_data)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, features=input_data, feature_names=[
        'energy_100g', 'fat_100g', 'sugars_100g', 'salt_100g', 'fiber_100g',
        'proteins_100g', 'bad_ingredient_count', 'additive_count', 'ingredients_length'
    ], plot_type="bar", show=False)
    st.pyplot(fig)

    # ========== Healthier Alternatives ==========
    st.subheader("💡 Healthier Alternatives")
    similar = df[df['health_label'] == 'Healthy']
    suggestions = similar.sample(3)[['product_name', 'nutriscore_score', 'health_label']]
    st.dataframe(suggestions.reset_index(drop=True))
