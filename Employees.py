import streamlit as st
import pandas as pd
from os.path import splitext

st.title('Streamlit - Employees')
sidebar = st.sidebar

DATA_URL = 'Employees.csv'

@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    return data
data = load_data()