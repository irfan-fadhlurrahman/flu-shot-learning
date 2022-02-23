import streamlit as st
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline

from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier

from train import impute

def load_pipeline():
    PATH = "/mnt/c/Project/flu-shot-learning/saved-pipeline/"
    with open(PATH + "saved_pipeline.bin", "rb") as file_out:
         pipeline_dict = pickle.load(file_out)
    return pipeline_dict

# backend
def predict_page():
    pipeline_dict = load_pipeline()

    st.title("Test aja dulu ngeload pipeline")
    st.write("""### we need your information""")

    age_group = st.selectbox("Age Group", pipeline_dict['features'].get('age_group'))
    education =	st.selectbox("Education", pipeline_dict['features'].get('education'))
    race = st.selectbox("Race", pipeline_dict['features'].get('race'))
    sex = st.selectbox("Sex", pipeline_dict['features'].get('sex'))
    income_poverty = st.selectbox("Income Poverty", pipeline_dict['features'].get('income_poverty'))

    ok = st.button("Calculate Probabilities of getting a vaccine")
    if ok:
        X = np.zeros((1, 35), int)
        X = pd.DataFrame(X, columns=list(pipeline_dict['features'].keys()))

        X.loc[0, 'age_group'] = age_group
        X.loc[0, 'education'] = education
        X.loc[0, 'race'] = race
        X.loc[0, 'sex'] = sex
        X.loc[0, 'income_poverty'] = income_poverty

        X = X.to_dict(orient='records')
        X = pipeline_dict['preprocessor'].transform(X)

        y_pred = pipeline_dict['model'].predict_proba(X)
        y_pred_h1n1 = y_pred[0][:, 1][0]
        y_pred_seasonal = y_pred[1][:, 1][0]

        st.subheader(f"The probability of getting the h1n1 vaccine: {y_pred_h1n1:.3f}")
        st.subheader(f"The probability of getting the seasonal vaccine: {y_pred_seasonal:.3f}")

def main():
    predict_page()

if __name__ == '__main__':
    main()

















#end of code
