import streamlit as st
import pandas as pd

st.title('Streamlit - Search Names')

DATA_URL = ('dataset.csv')

@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    return data


@st.cache
def load_data_bysex(sex):
    data = pd.read_csv(DATA_URL)
    filtered_data_bysex = data[data['sex'] == sex]

    return filtered_data_bysex

data = load_data()
selected_sex = st.selectbox('Select Sex ', data['sex'].unique())
btnfilterbysex = st.button('Filtrar por sexo')
if (btnfilterbysex):
    filterbysex = load_data_bysex(selected_sex)
    count_row = filterbysex.shape[0]
    st.write(f"Nombres Totales: {count_row}")

    st.dataframe(filterbysex)