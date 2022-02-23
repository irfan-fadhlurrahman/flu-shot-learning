import streamlit as st
import pickle
import numpy as np
import pandas as pd
#from train import impute
from predict import load_pipeline, predict_page

if __name__ == '__main__':
    predict_page()
