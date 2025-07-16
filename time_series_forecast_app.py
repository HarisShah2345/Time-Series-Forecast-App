import streamlit as st
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Time-Series Forecaster", layout="wide")
st.title("\U0001F4C8 Time-Series Forecasting App")

st.sidebar.header("Configuration")
forecast_period = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.write(data.head())

    if 'ds' not in data.columns or 'y' not in data.columns:
        st.error("The CSV must have two columns: 'ds' (date) and 'y' (value to forecast).")
    else:
        data['ds'] = pd.to_datetime(data['ds'])

        with st.spinner("Training Prophet model..."):
            model = Prophet()
            model.fit(data)

        future = model.make_future_dataframe(periods=forecast_period)
        forecast = model.predict(future)

        st.subheader("Forecast Plot")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], name='Actual'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecasted Values")
        forecast_tail = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_period)
        st.write(forecast_tail)

        csv = forecast_tail.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast CSV",
            data=csv,
            file_name='forecast.csv',
            mime='text/csv'
        )
else:
    st.info("Please upload a CSV file with 'ds' and 'y' columns to get started.")
