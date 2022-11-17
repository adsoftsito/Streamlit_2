import streamlit as st
import pandas as pd
import numpy as np
import datetime
# Crear el título para la aplicación web
st.title("Mi Primera App con Streamlit")
sidebar = st.sidebar
sidebar.title("Esta es la barra lateral.")
sidebar.write("Aquí van los elementos de entrada.")
st.header("Información sobre el Conjunto de Datos")
st.header("Descripción de los datos ")
st.write("""
Este es un simple ejemplo de una app para predecir
¡Esta app predice mis datos!
""")

if sidebar.checkbox('Mostrar Dataframe'):
    chart_data = pd.read_csv('https://raw.githubusercontent.com/jeaggo/tc3068/master/Superstore.csv')
    st.write(chart_data)

# Give user the current date
today = datetime.date.today()
today_date = st.date_input('Current date', today)
st.success('Current date: `%s`' % (today_date))

