import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import json
from branca.colormap import LinearColormap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Region configurations and data loading functions remain the same...

# Data loading
df = load_main_data()
geojson_data = load_geojson()
forecast_df = load_forecast()
metrics_df = load_metrics()

# Sidebar filters (remain the same...)

# Header
st.title("📈 Ghana Infectious Disease Trends Dashboard")
st.markdown("#### Machine Learning-Powered Epidemiology | HIV/AIDS Focus")
st.markdown("---")

# Section 1: Time Series (with improved checks)
st.subheader("1. National Disease Trends Over Time")
if not selected_disease:
    st.warning("Please select at least one disease to display trends.")
else:
    if df_time.empty:
        st.warning("No data available for the selected filters.")
    else:
        fig1 = px.line(df_time, x='date', y=selected_disease, color='region')
        fig1.update_layout(width=1200, height=600)
        st.plotly_chart(fig1, use_container_width=True)

# Section 2: Choropleth Map (with disease check)
st.subheader("2. Regional Distribution Map (10 Original Regions)")
if not df_single.empty and selected_disease:
    latest = df_single.groupby('region').last().reset_index().rename(columns={'region':'Region'})
    if not latest.empty:
        m = folium.Map(location=[7.9, -1.0], zoom_start=6, tiles="Stamen Toner")
        folium.Choropleth(
            geo_data=geojson_data,
            data=latest,
            columns=['Region', selected_disease[0]],
            key_on='feature.properties.shapeName',
            fill_color='YlGnBu',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=selected_disease[0].replace('_', ' ').title(),
            nan_fill_color='gray',
            tooltip=folium.GeoJsonTooltip(
                fields=['shapeName', selected_disease[0]],
                aliases=['Region', selected_disease[0].replace('_', ' ').title()]
            )
        ).add_to(m)
        st_folium(m, width=500, height=900)
    else:
        st.warning("No recent data available for selected regions and disease.")
elif not selected_disease:
    st.warning("Please select at least one disease to display the map.")
else:
    st.warning("No data available for the selected date range and regions.")

# Section 3: Behavioral Correlation (remains same...)

# Section 4: Correlation Heatmap (remains same...)

# NEW Section 5: Forecasts
st.subheader("5. Disease Incidence Forecasts (2030)")
if not forecast_df.empty:
    fig5 = px.bar(forecast_df, x='region', y='predicted_2030', color='disease',
                  barmode='group', title='Projected 2030 Disease Incidence by Region')
    fig5.update_layout(xaxis_title='Region', yaxis_title='Predicted Incidence Rate')
    st.plotly_chart(fig5, use_container_width=True)
else:
    st.warning("Forecast data not available.")

# Section 6 & 7: Model Performance (remains same...)

# Footer
st.markdown("---")
st.markdown("*Developed by Valentine Ghanem | MSc Public Health & Data Science*")
