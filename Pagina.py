import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

st.title('Fractal solutions')

df = ('Monterrey_dataset.csv')
df = pd.read_csv(df)


opcion_hotel = df['venue_code'].unique().tolist()
hotel = st.selectbox('¿Que hotel quisiera ver?', opcion_hotel,0)
df = df[df['venue_code']==hotel]

fig = px.scatter(df,x='room_revenue',y='rooms_occupied',size='occ',color='occ',
hover_name = 'occ')

fig.update_layout(width=400)

st.write(fig)

def report_metric(pred, test, model_name):
     # Creates report with mae, rmse and r2 metric and returns as df
     mae = mean_absolute_error(pred, test)
     mse = mean_squared_error(pred, test)
     rmse = np.sqrt(mse)
     r2 = r2_score(test, pred)
     metric_data = {'Metric': ['MAE', 'RMSE', 'R2'], model_name: [mae, rmse, r2]}
     metric_df = pd.DataFrame(metric_data)
     return metric_df
def plot_preds(data_date,test_date, target, pred):
     # Plots prediction vs real 
     fig = plt.figure(figsize=(20,10))
     plt.plot(data_date, target, label = 'Real')
     plt.plot(test_date, pred, label = 'Pred')
     plt.legend()
     st.pyplot(fig)

test_period = -20
test = df[test_period:]
train = df[:test_period]

x1_test = test[['rooms_available','hotelSearchesMX','is_holiday','hocc','dayofweek','day','venue_encode']]
y1_test = test[['occ']]
x1_train = train[['rooms_available','hotelSearchesMX','is_holiday','hocc','dayofweek','day','venue_encode']]
y1_train = train[['occ']]

#'date', 'venue_code', 'found', 'rooms_available', 'rooms_occupied',
       #'room_revenue', 'state', 'hotelSearchesMX', 'is_holiday', 'hocc',
       #'hotelSearches', 'dayofweek', 'day', 'month', 'venue_encode', 'occ',
       #'Evento', 'Lugar', 'Modalidad'

lr = LinearRegression()
lr.fit(x1_train, y1_train)
m1pred = lr.predict(x1_test)

metric1 = report_metric(m1pred, y1_test, "Linear Regression")



st.title("Model 1: ")
st.write("Model 1 works with linear regression as base model.")
st.write("The columns it used are: col1, col2, col3, day_of_week, day_of_month, month, week_of_year, season")
st.write(metric1)
"""
### Real vs Pred. Plot for 1. Model
"""
plot_preds(df["date"],test["date"], df["occ"], m1pred)