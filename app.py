import streamlit as st
import joblib
import pandas as pd
import json
import numpy as np

# Page configuration
st.set_page_config(page_title="Weekly Sales Predictor", layout="wide")

# Load model and preprocessing objects
@st.cache_resource
def load_model_and_objects():
    model = joblib.load('xgboost_model.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('feature_names.json', 'r') as f:
        feature_names = json.load(f)
    return model, scaler, feature_names

model, scaler, feature_names = load_model_and_objects()

# App title and description
st.title('📈 Weekly Sales Prediction App')
st.markdown("""
This app predicts weekly sales based on input features.
Enter the values for each feature and click 'Predict' to get the forecasted weekly sales.
""")

# Create input form
st.header('Input Features')
input_data = {}

# Create columns for better layout
col1, col2 = st.columns(2)

# Split features between two columns
for i, feature in enumerate(feature_names):
    if i % 2 == 0:
        input_data[feature] = col1.number_input(feature, value=0.0)
    else:
        input_data[feature] = col2.number_input(feature, value=0.0)

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Prediction section
st.header('Prediction')
if st.button('🔮 Predict Weekly Sales', key='predict'):
    try:
        # Preprocess input data
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)

        # Display prediction with formatting
        st.success(f"""
        ### Predicted Weekly Sales: **${prediction[0]:,.2f}**

        This prediction is based on the input features you provided.
        """)

        # Show feature importance (optional)
        st.subheader('Feature Importance')
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        st.bar_chart(importance_df.set_index('Feature'))

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.warning("Please check your input values and try again.")

# Add a section for batch prediction
st.header('Batch Prediction (CSV Upload)')
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)

        # Check if all required features are present
        missing_features = set(feature_names) - set(batch_data.columns)
        if missing_features:
            st.error(f"Missing features in uploaded file: {', '.join(missing_features)}")
        else:
            # Select only the required features
            batch_data = batch_data[feature_names]

            # Preprocess and predict
            batch_scaled = scaler.transform(batch_data)
            batch_predictions = model.predict(batch_scaled)

            # Add predictions to DataFrame
            batch_data['Predicted_Weekly_Sales'] = batch_predictions

            # Show results
            st.success(f"Successfully predicted {len(batch_data)} records!")
            st.dataframe(batch_data)

            # Offer download
            csv = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name='sales_predictions.csv',
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Add some information about the model
st.sidebar.header("About the Model")
st.sidebar.markdown("""
- **Model Type**: XGBoost Regressor
- **Training Data**: Historical sales data with multiple features
- **Performance**: Optimized using RandomizedSearchCV
- **Use Case**: Predict weekly sales for retail stores
""")

st.sidebar.header("How to Use")
st.sidebar.markdown("""
1. Enter values for all features
2. Click 'Predict Weekly Sales' button
3. View the prediction and feature importance
4. For batch predictions, upload a CSV file
""")
