import streamlit as st
import pandas as pd

st.title('Streamlit con atributo cache')

Data_Url = ('dataset.csv')

@st.cache
def load_data(nrows):
    data = pd.read_csv(Data_Url,nrows=nrows)
    return data

data_load_state = st.text('Loading data...')

data = load_data(10)

data_load_state.text("Done: (Using st.cache)")

st.dataframe(data)