import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import seaborn as sns
import matplotlib.pyplot as plt
import json

# Mapping for new to old regions
region_mapping = {
    'Ahafo': 'Brong-Ahafo',
    'Bono': 'Brong-Ahafo',
    'Bono East': 'Brong-Ahafo',
    'Savannah': 'Northern',
    'North East': 'Northern',
    'Western North': 'Western',
    'Oti': 'Volta'
}

# Original 10 regions
original_regions = [
    'UPPER WEST', 'UPPER EAST', 'NORTHERN', 'BRONG-AHAFO',
    'ASHANTI', 'EASTERN', 'WESTERN', 'CENTRAL', 'GREATER ACCRA', 'VOLTA'
]

# Load datasets
@st.cache_data
def load_main_data():
    df = pd.read_csv("ghana_infectious_disease_model_dataset_cleaned.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['region'] = df['region'].replace(region_mapping).str.upper()
    # Keep only original regions
    df = df[df['region'].isin(original_regions)]
    return df

@st.cache_data
def load_geojson():
    with open("geoBoundaries-GHA-ADM1_simplified.geojson", "r") as f:
        gj = json.load(f)
        # Filter features to original regions
        gj['features'] = [feat for feat in gj['features']
                          if feat['properties']['shapeName'].upper() in original_regions]
        # Standardize shapeName property
        for feat in gj['features']:
            name = feat['properties']['shapeName']
            # Map if needed then uppercase
            feat['properties']['shapeName'] = region_mapping.get(name, name).upper()
        return gj

@st.cache_data
def load_forecast():
    df = pd.read_csv("hiv_predicted_2030_by_region.csv")
    df['region'] = df['region'].replace(region_mapping).str.upper()
    return df

@st.cache_data
def load_metrics():
    return pd.read_csv("model_performance_metrics.csv")

# Load data
df = load_main_data()
geojson_data = load_geojson()
forecast_df = load_forecast()
metrics_df = load_metrics()

# Sidebar filters
st.sidebar.header("Filter Panel")
all_regions = df['region'].unique().tolist()
select_all = st.sidebar.checkbox("Select all regions", True)
selected_region = st.sidebar.multiselect(
    "Regions", all_regions,
    default=all_regions if select_all else []
)

disease_opts = ['hiv_incidence', 'malaria_incidence', 'tb_incidence']
selected_disease = st.sidebar.selectbox("Disease", disease_opts)

# Date filters
min_date = df['date'].dt.date.min()
max_date = df['date'].dt.date.max()
# Date range for trends
date_range = st.sidebar.date_input(
    "Select Date Range", [min_date, max_date]
)
# Single date for map and scatter
selected_date = st.sidebar.date_input(
    "Select Date for Map/Scatter", min_value=min_date,
    max_value=max_date, value=min_date
)

# Data slices
# For time series and heatmap (range)
df_time = df[(df['region'].isin(selected_region)) &
             (df['date'].dt.date >= date_range[0]) &
             (df['date'].dt.date <= date_range[1])]
# For map and scatter (single date)
df_single = df[(df['region'].isin(selected_region)) &
               (df['date'].dt.date == selected_date)]

# Title and branding
st.title("📈 Ghana Infectious Disease Trends Dashboard")
st.markdown("#### Machine Learning-Powered Epidemiology | HIV/AIDS Focus")
st.markdown("---")

# Section 1: Time Series (uses df_time)
st.subheader("1. National Disease Trends Over Time")
fig1 = px.line(
    df_time, x='date', y=selected_disease, color='region',
    title="Trends Over Time"
)
fig1.update_layout(width=1200, height=600)
st.plotly_chart(fig1, use_container_width=True)

# Section 2: Choropleth Map (uses df_single)
st.subheader("2. Regional Distribution Map (10 Original Regions)")
latest = df_single.groupby('region').last().reset_index().rename(columns={'region':'Region'})

m = folium.Map(location=[7.9, -1.0], zoom_start=6, tiles="CartoDB positron")
folium.Choropleth(
    geo_data=geojson_data,
    data=latest,
    columns=['Region', selected_disease],
    key_on='feature.properties.shapeName',
    fill_color='YlGnBu',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name=selected_disease.replace('_',' ').title(),
    nan_fill_color='gray'
).add_to(m)
folium.GeoJsonTooltip(fields=['shapeName']).add_to(m)
st_folium(m, width=700, height=500)

# Section 3: Behavioral & Demographic Correlation (uses df_single)
st.subheader("3. Behavioral & Demographic Correlation")
var = st.selectbox(
    "Compare with",
    ['education_access_index','condom_use_rate','urbanization_level',
     'hiv_awareness_index','youth_unemployment_rate']
)
fig2 = px.scatter(
    df_single, x=var, y=selected_disease, color='region',
    title=f"{var.replace('_',' ').title()} vs {selected_disease.replace('_',' ').title()}"
)
st.plotly_chart(fig2, use_container_width=True)

# Section 4: Correlation Heatmap (uses df_time)
st.subheader("4. Correlation Heatmap")
num_cols = [
    'hiv_incidence','malaria_incidence','tb_incidence',
    'education_access_index','condom_use_rate','urbanization_level',
    'hiv_awareness_index','youth_unemployment_rate'
]
corr = df_time[num_cols].corr()
fig_hm, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='viridis', ax=ax)
st.pyplot(fig_hm)

# Section 5: Forecasts (National level)
st.subheader("5. ML Forecasting (National)")
if {'year','hiv_predicted'}.issubset(forecast_df.columns):
    fig3 = px.line(
        forecast_df, x='year', y='hiv_predicted',
        title='National HIV Forecast to 2030'
    )
    fig3.update_layout(height=500)
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("Forecast data missing.")

# Section 6: Model Performance Summary
st.subheader("6. Model Performance Summary")
st.dataframe(metrics_df, use_container_width=True)

# Section 7: Model Metrics Correlation Heatmap
st.subheader("7. Model Metrics Correlation Heatmap")
metrics_numeric = metrics_df.select_dtypes(include='number')
if not metrics_numeric.empty:
    corr_metrics = metrics_numeric.corr()
    fig_mm, ax_mm = plt.subplots(figsize=(8,6))
    sns.heatmap(corr_metrics, annot=True, fmt='.2f', cmap='coolwarm', ax=ax_mm)
    st.pyplot(fig_mm)
else:
    st.warning("No numeric metrics to correlate.")

# Footer
st.markdown("---")
st.markdown("*Developed by Valentine Ghanem | MSc Public Health & Data Science*")
