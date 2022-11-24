import streamlit as st
import pandas as pd

DATA_URL = 'uber-raw-data-sep14.csv'

st.title("Viajes UBER")

st.header("Viajes de Uber en la ciudad de Nueva Yorkcon filtros por hora.")



@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data['date/time'] = pd.to_datetime(data['date/time'])
    data['time_hour'] = pd.to_datetime(data['date/time']).dt.hour
   
    return data

data = load_data(1000)

Horas = data['time_hour'].unique().tolist()
hora = st.selectbox('¿Que hora de viajes quieres ver?', Horas,0)
data = data[data['time_hour']==hora]


st.dataframe(data)
st.map(data)