import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

# Streamlit UI
st.title("🌾 Smart Crop Recommendation with Explainability")
st.write("This dashboard predicts the most suitable crop and explains the decision using SHAP and LIME.")

# Load and preprocess data
df = pd.read_csv("Crop_recommendation.csv")
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[features].values
y = df['label'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_seq = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_cat, test_size=0.2, random_state=42)
X_flat_train = X_train[:, 0, :]
X_flat_test = X_test[:, 0, :]

# Build and train LSTM
@st.cache_resource
def train_model():
    model = Sequential([
        Input(shape=(1, len(features))),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(y_cat.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=15, validation_split=0.2, verbose=0)
    return model

model = train_model()

def model_predict(x):
    x_reshaped = x.reshape((x.shape[0], 1, x.shape[1]))
    return model.predict(x_reshaped)

# Input sliders
st.sidebar.header("Enter Soil and Weather Data")
user_input = {
    feature: st.sidebar.slider(feature, float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
    for feature in features
}

input_array = np.array([list(user_input.values())])
input_scaled = scaler.transform(input_array)
input_seq = input_scaled.reshape((1, 1, len(features)))

# Prediction
prediction = model.predict(input_seq)
predicted_crop = le.inverse_transform([np.argmax(prediction)])[0]
st.subheader("🧠 Predicted Crop:")
st.success(f"Recommended Crop: **{predicted_crop}**")

# SHAP Explanation
st.subheader("🔍 SHAP Feature Importance")

with st.spinner("Generating SHAP explanation..."):
    explainer = shap.KernelExplainer(model_predict, X_flat_train[:100])
    shap_values = explainer.shap_values(input_scaled)
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap.Explanation(values=shap_values[np.argmax(prediction)][0],
                                         base_values=explainer.expected_value[np.argmax(prediction)],
                                         data=input_scaled[0],
                                         feature_names=features), max_display=7)
    st.pyplot(fig)

# LIME Explanation
st.subheader("🌈 LIME Explanation")

with st.spinner("Generating LIME explanation..."):
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_flat_train,
        feature_names=features,
        class_names=le.classes_,
        mode='classification'
    )
    exp = explainer_lime.explain_instance(input_scaled[0], model_predict, num_features=7)
    st.components.v1.html(exp.as_html(), height=600)

# Footer
st.markdown("---")
st.caption("Built with ❤️ for precision agriculture and explainable AI.")
