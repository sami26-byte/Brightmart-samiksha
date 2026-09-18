import pickle

import pandas as pd
import streamlit as st

MODEL_PATH = 'linear_reg (1).sav'


@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as model_file:
        return pickle.load(model_file)


model = load_model()

st.title('Sales Prediction App')

TV = st.number_input('TV Advertising Budget', min_value=0.0, value=0.0)
Radio = st.number_input('Radio Advertising Budget', min_value=0.0, value=0.0)
Newspaper = st.number_input('Newspaper Advertising Budget', min_value=0.0, value=0.0)

if st.button('Predict Sales'):
    try:
        input_data = pd.DataFrame(
            [[TV, Radio, Newspaper]],
            columns=['TV', 'Radio', 'Newspaper'],
            dtype=float,
        )
        prediction = model.predict(input_data)[0]
        st.success(f'Predicted Sales: {prediction:.2f}')
    except Exception as exc:
        st.error(f'Prediction failed: {exc}')
