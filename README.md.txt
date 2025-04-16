# 🍎 Food Healthiness Scoring System

A smart ML-powered Streamlit app that predicts how healthy packaged food is — based on nutritional values, ingredients, and additives. Built with real data from OpenFoodFacts.

---

## 🚀 Live Demo
[Click here to try the app →](https://share.streamlit.io/your-username/food-healthiness-score-app/main/app.py)

*(Replace with your actual Streamlit Cloud link after deployment)*

---

## 📦 Features

✅ Predicts a **Health Label** (`Healthy`, `Moderate`, `Unhealthy`) using classification  
✅ Predicts the actual **NutriScore (0–40+)** using regression  
✅ Works with:
- Manual nutrition input
- Barcode number input
- 📤 Barcode image upload *(Streamlit Cloud-compatible)*
✅ 💡 SHAP-based feature explanation
✅ 🔁 Recommends healthier alternative products

---

## 📥 How to Use

### 🧾 Option 1: Manual Entry
Enter nutrition facts and ingredient info to get predictions.

### 📷 Option 2: Enter Barcode
Paste the barcode number from a packaged food item to auto-fetch from OpenFoodFacts.

### 🖼️ Option 3: Upload Barcode Image
Snap a picture of a barcode, upload it, and let the app scan + fetch info for you.

> ✅ Works great on both desktop and mobile!

---

## 📸 Screenshots

| Input | Prediction + SHAP | Healthier Suggestions |
|-------|-------------------|------------------------|
| ![form](screenshots/form.png) | ![result](screenshots/prediction.png) | ![alt](screenshots/alternatives.png) |

*(You can add your own screenshots or replace these)*

---

## 🛠️ Setup Locally

```bash
# Clone repo
https://github.com/your-username/food-healthiness-score-app
cd food-healthiness-score-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
```

---

## 🧠 Tech Stack
- Python, Pandas, Scikit-learn
- Streamlit, Streamlit WebRTC
- OpenCV + Pyzbar for barcode detection
- SHAP for model explainability

---

## 📂 Folder Structure
```
├── app/
│   └── app.py
├── models/
│   ├── classifier.pkl
│   └── regressor.pkl
├── data/
│   └── engineered_food_data.csv
├── requirements.txt
└── README.md
```

---

## 🙌 Credits
- [OpenFoodFacts.org](https://world.openfoodfacts.org/) for open barcode & nutrition data
- Streamlit + SHAP for building powerful explainable apps

---

## 📢 License
This project is MIT-licensed. Feel free to remix, reuse, and improve! 🎉
