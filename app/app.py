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
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import gdown

# ========== Download Models from Google Drive ==========
def download_from_gdrive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

os.makedirs("models", exist_ok=True)
download_from_gdrive("1t2bQyTOOT9IaHXk0nhANl9G8og5wInlf", "models/classifier.pkl")
download_from_gdrive("16n3eWgKBqsImhbsgSJGQCwN1UMxnFkhW", "models/regressor.pkl")

# ========== Load Models ==========
clf = joblib.load("models/classifier.pkl")
reg = joblib.load("models/regressor.pkl")

# ========== Load Dataset for Alternatives ==========
df = pd.read_csv("data/engineered_food_data.csv")

# ========== Streamlit App Config ==========
st.set_page_config(page_title="🍎 Food Healthiness Scorer", layout="centered")
st.title("🍎 Food Healthiness Scoring System")
st.markdown("Enter nutrition facts or scan/upload barcode to evaluate packaged food healthiness!")

# ========== Barcode Input ==========
st.subheader("📷 Barcode Input Options")
barcode = st.text_input("🔢 Enter barcode manually")

# ========== Barcode Scanner Class ==========
if 'detected_barcode' not in st.session_state:
    st.session_state.detected_barcode = ''

class BarcodeScanner(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        barcodes = pyzbar.decode(img)
        for barcode in barcodes:
            barcode_data = barcode.data.decode("utf-8")
            st.session_state.detected_barcode = barcode_data
        return img

# ========== Side-by-side: Live Scanner + Image Upload ==========
st.markdown("### 🔍 Scan or Upload Barcode")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🎥 Live Barcode Scanner")
    webrtc_streamer(key="barcode", video_transformer_factory=BarcodeScanner)
    if st.session_state.get("detected_barcode"):
        st.success(f"📸 Detected from webcam: {st.session_state.detected_barcode}")
        barcode = st.session_state.detected_barcode

with col2:
    st.markdown("#### 🖼️ Upload Barcode Image")
    uploaded = st.file_uploader("Upload barcode image")
    if uploaded:
        if uploaded.type.startswith("image/"):
            try:
                image = Image.open(uploaded)
                decoded = pyzbar.decode(image)
                if decoded:
                    scanned_code = decoded[0].data.decode("utf-8")
                    st.success(f"🧾 Detected from image: {scanned_code}")
                    barcode = scanned_code
                else:
                    st.warning("⚠️ No barcode detected in the image.")
            except Exception as e:
                st.error(f"❌ Failed to process image: {e}")
        else:
            st.error("❌ Please upload a valid image file (JPEG, PNG, etc.)")

# ✅ Show active barcode
if barcode:
    st.write(f"🔎 Using barcode: `{barcode}`")

# ========== Barcode API Lookup ==========
if barcode:
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
    res = requests.get(url)

    if res.status_code == 200:
        data = res.json()
        if data.get("status") == 1:
            product = data["product"]
            product_name = product.get("product_name", "Unknown Product")
            product_image = product.get("image_url", "")

            st.success(f"✅ Found Product: {product_name}")
            if product_image:
                st.image(product_image, caption=product_name, width=200)

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

            # Warn if data is missing
            if sum(input_features[:6]) == 0:
                st.warning("⚠️ Product has missing nutrition info. Prediction may be inaccurate.")

            # Display as table
            st.markdown("### 🧮 Features Extracted from Product:")
            feature_names = [
                'Energy (kcal per 100g)',
                'Fat (g per 100g)',
                'Sugars (g per 100g)',
                'Salt (g per 100g)',
                'Fiber (g per 100g)',
                'Proteins (g per 100g)',
                'Bad Ingredient Count',
                'Additives Count',
                'Ingredients Length'
            ]
            features_df = pd.DataFrame({"Feature": feature_names, "Value": input_features})
            st.table(features_df)

            # Predictions
            pred_label = clf.predict([input_features])[0]
            pred_score = reg.predict([input_features])[0]

            st.markdown("### 🧠 Model Predictions")
            st.success(f"🏷️ **Health Label:** {pred_label}")
            st.success(f"📉 **NutriScore:** {pred_score:.2f}")
        else:
            st.error("❌ Product not found in OpenFoodFacts database.")
    else:
        st.error("❌ Failed to fetch data from API.")

# ========== Manual Input Dropdown with Styled Header ==========
st.markdown("### 📝 Manual Nutrition Input")
with st.expander("", expanded=False):
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

# ========== Prediction ==========
if submitted:
    input_data = np.array([[energy, fat, sugar, salt, fiber, protein,
                            bad_ingredients, additives, ingredients_len]])

    pred_label = clf.predict(input_data)[0]
    pred_score = reg.predict(input_data)[0]

    st.subheader("🧠 Predictions from Manual Input")
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
