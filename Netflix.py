import streamlit as st
import pandas as pd
from os.path import splitext

st.title('Streamlit - Netflix')
sidebar = st.sidebar

DATA_URL = 'movies.csv'

@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    return data
data = load_data()

st.header("Dataset")
agree = sidebar.checkbox("Quieres ver la base de datos ? ")
if agree:
    st.dataframe(data)

@st.cache
def load_data_byname(name):
    data = pd.read_csv(DATA_URL)
    filtered_data_byname = data[data['name'].str.upper().str.contains(name)]

    return filtered_data_byname

myname = sidebar.text_input('Nombre de la pelicula: ')
btnname = sidebar.button('Buscar por nombre')
if (btnname):
    filterbyname = load_data_byname(myname.upper())
    count_row = filterbyname.shape[0]
    st.write(f"Peliculas Totales: {count_row}")

    st.dataframe(filterbyname)

@st.cache
def load_data_bydirector(director):
    data = pd.read_csv(DATA_URL)
    filtered_data_bydirector = data[data['director'].str.contains(director)]

    return filtered_data_bydirector


director = sidebar.selectbox('Nombre del Director: ',data['director'].unique())
btndirector = st.sidebar.button('Buscar por Director')

if (btndirector):
    filterbydirector = load_data_bydirector(director)
    count_row = filterbydirector.shape[0]
    st.write(f"Peliculas Totales: {count_row}")

    st.dataframe(filterbydirector)