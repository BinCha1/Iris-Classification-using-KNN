import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
with open('knn_iris_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler_iris.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Page config
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    body {
        background-color: #ffffff;
    }
    .main-header {
        text-align: center;
        color: #2E8B57;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #555;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #43cea2, #185a9d);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .stButton > button {
        background: linear-gradient(135deg, #43cea2, #185a9d);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.25);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌸 Iris Flower Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by K-Nearest Neighbors (KNN) Algorithm</p>', unsafe_allow_html=True)

# Feature Input Section

st.markdown("### 🌿 Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)

with col2:
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

st.markdown('</div>', unsafe_allow_html=True)

# Prediction Section

st.markdown("### 🔎 Classification")

if st.button("🌸 Predict Iris Species"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    species_emoji = {
        'setosa': '🌻',
        'versicolor': '🌷',
        'virginica': '🌺'
    }
    emoji = species_emoji.get(prediction.lower(), '🌸')

    st.markdown(f'''
    <div class="prediction-box">
        {emoji} The predicted Iris species is:<br>
        <span style="font-size: 1.5rem;">{prediction.capitalize()}</span>
    </div>
    ''', unsafe_allow_html=True)

    if prediction.lower() == 'setosa':
        st.info("🌻 **Iris Setosa** - Known for its small size and distinctive features!")
    elif prediction.lower() == 'versicolor':
        st.info("🌷 **Iris Versicolor** - A beautiful medium-sized iris variety!")
    else:
        st.info("🌺 **Iris Virginica** - The largest of the three iris species!")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("💻 *Built with Streamlit & scikit-learn* ")
