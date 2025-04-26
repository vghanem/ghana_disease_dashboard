import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import json
import numpy as np
from branca.colormap import LinearColormap

# Data loading functions
@st.cache_data
def load_main_data():
    df = pd.read_csv("ghana_infectious_disease_model_dataset_cleaned.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['region'] = df['region'].str.upper().str.strip()
    return df

@st.cache_data
def load_geojson():
    with open("geoBoundaries-GHA-ADM1_simplified.geojson") as f:
        gj = json.load(f)
        valid_features = []
        for feature in gj['features']:
            original_name = feature['properties']['shapeName'].title()
            if original_name in df['region'].values:
                feature['properties']['shapeName'] = original_name
                valid_features.append(feature)
        gj['features'] = valid_features
        return gj

@st.cache_data
def load_forecast():
    df = pd.read_csv("hiv_predicted_2030_by_region.csv")
    df['region'] = df['region'].str.upper().str.strip()
    return df

# Data loading
df = load_main_data()
geojson_data = load_geojson()
forecast_df = load_forecast()

# Sidebar filters
st.sidebar.header("Filter Panel")
selected_regions = st.sidebar.multiselect(
    "Regions", df['region'].unique(), default=df['region'].unique()
)

disease_opts = ['hiv_incidence', 'malaria_incidence', 'tb_incidence']
selected_diseases = st.sidebar.multiselect(
    "Disease", disease_opts, default=disease_opts
)

min_date, max_date = df['date'].min().date(), df['date'].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

# Data filtering
df_time = df[(df['region'].isin(selected_regions)) &
             (df['date'].dt.date >= date_range[0]) &
             (df['date'].dt.date <= date_range[1])]

# Dashboard header
st.title("📈 Ghana Infectious Disease Trends Dashboard")
st.markdown("#### Machine Learning-Powered Epidemiology | HIV/AIDS Focus")
st.markdown("---")

# Section 1: Time Series
st.subheader("1. National Disease Trends Over Time")
if not selected_diseases:
    st.warning("Please select at least one disease to display trends.")
else:
    if df_time.empty:
        st.warning("No data available for selected filters.")
    else:
        fig1 = px.line(df_time, x='date', y=selected_diseases, color='region')
        fig1.update_layout(
            width=1500, height=600,
            showlegend=True,
            legend=dict(
                x=1.05, y=1, traceorder='normal', orientation='v',
                font=dict(size=10), bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='Black', borderwidth=1
            )
        )
        st.plotly_chart(fig1, use_container_width=True)

# Section 5: Forecasts
st.subheader("5. Disease Incidence Forecasts (2030)")
if not forecast_df.empty:
    fig5 = px.bar(forecast_df, x='region', y='predicted_2030', color='region',
                  title='Projected 2030 Disease Incidence by Region')
    fig5.update_layout(xaxis_title='Region', yaxis_title='Predicted Incidence Rate')
    st.plotly_chart(fig5, use_container_width=True)
else:
    st.warning("Forecast data not available.")
