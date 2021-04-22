import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shap
from PIL import Image
from catboost import CatBoostRegressor
shap.initjs()

this_dir = os.path.abspath(os.path.dirname(__file__))
save_dir = os.path.join(this_dir, 'saves')


@st.cache()
def load_model():
    model = CatBoostRegressor().load_model(os.path.join(save_dir,
                                                        'catboost_bikedemand_model_rmsle_grid.cbm'))
    return model


@st.cache()
def prepare_shap(model: object, data: pd.DataFrame) -> object:
    'Returns shapley values of given input data'
    explainer = shap.Explainer(model, feature_perturbation="tree_path_dependent")
    shap_val = explainer.shap_values(data)
    return shap_val, explainer


def prediction_maker(model: object, test_data: pd.DataFrame) -> np.array:
    'makes prediction with negative values corrected to zero'

    pred = model.predict(test_data)
    # all negative predictions need to be pulled up
    pred = [i if i > 0 else 0 for i in pred]
    return pred


st.title('Welcome to The Bike Demand Forecast App!')
shap_beeswarm = Image.open(os.path.join(save_dir, 'shap_plot.png'))
st.image(shap_beeswarm, caption='Shapley Values present: The Factors influence on a bikes demand!')
st.title('Please, choose required conditions on the sidebar, and submit for prediction.')


with st.sidebar:
    temp = st.slider('Choose temperature [C]', 0, 40, value=20)
    humidity = st.slider('Choose humidity [%]', 20, 100, value=50)
    y_help = 'Important! The model is based on data from 2011 - 2012, \
             therefore a forecast for next years may be slightly biased'
    st.write(y_help)
    year = st.slider('Year', 2011, 2021, value=2011, step=1)
    season = st.selectbox('Select a season', ('Spring', 'Summer', 'Fall', 'Winter'), 0)
    month = st.selectbox('Select a month', ('January', 'February', 'March', 'April',
                                            'May', 'June', 'July', 'August', 'September', 'October',
                                            'Novemver', 'December'), 3)
    workingday = st.selectbox('Is it a working day?', ('Yes', 'No'), 0)
    hour = st.slider('Hour (system 24)', 0, 24, 16, step=1)
    weather = st.slider('How good is the weather? 1-best 4-worst', 1, 4, value=1, step=1)
    submit = st.button('SUBMIT')

seasons_dict = {'Spring': 1, 'Summer': 2, 'Fall': 3, 'Winter': 4}
month_dict = {'January': 1, 'February': 2, 'March': 3, 'April': 4,
              'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10,
              'Novemver': 11, 'December': 12}
workingday_dict = {'Yes': 1, 'No': 0}

data = {'season': [seasons_dict[season]], 'workingday': [workingday_dict[workingday]],
        'weather': [int(weather)], 'temp': [temp], 'humidity': [humidity], 'year': [year],
        'month': [month_dict[month]], 'hour': [hour]}
data = pd.DataFrame.from_dict(data, orient='columns')
model = load_model()
shap_val, explainer = prepare_shap(model, data)
explanation = shap.Explanation(values=shap_val[0],
                               base_values=explainer.expected_value,
                               data=data.iloc[0, :],
                               feature_names=data.columns.tolist())


if submit:
    pred = prediction_maker(model, data)
    st.write(f'Predicted demand is: {int(np.round(pred[0]))}')
    st.write('Below, there is an explanation on how this prediction is influenced by given factors')
    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation,
                         show=False)
    st.pyplot(fig)
