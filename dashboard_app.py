import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import json

# Load main dataset
@st.cache_data
def load_main_data():
    df = pd.read_csv("ghana_infectious_disease_model_dataset_cleaned.csv")
    df['date'] = pd.to_datetime(df['date'])

    # Normalize to 10 original Ghana regions (pre-2018 structure)
    region_mapping = {
        'Ahafo': 'Brong-Ahafo',
        'Bono': 'Brong-Ahafo',
        'Bono East': 'Brong-Ahafo',
        'Savannah': 'Northern',
        'North East': 'Northern',
        'Western North': 'Western',
        'Oti': 'Volta'
    }
    df['region'] = df['region'].replace(region_mapping)
    return df

# Load GeoJSON
@st.cache_data
def load_geojson():
    with open("geoBoundaries-GHA-ADM1_simplified.geojson", "r") as f:
        return json.load(f)

# Load ML forecast results
@st.cache_data
def load_forecast():
    return pd.read_csv("hiv_predicted_2030_by_region.csv")

# Load model performance metrics
@st.cache_data
def load_metrics():
    return pd.read_csv("model_performance_metrics.csv")

# Load all data
df = load_main_data()
geojson_data = load_geojson()
forecast_df = load_forecast()
metrics_df = load_metrics()

# Sidebar filters
st.sidebar.header("Filter Panel")
selected_region = st.sidebar.multiselect("Select Region(s):", df['region'].unique(), default=df['region'].unique())
selected_disease = st.sidebar.selectbox("Disease", ['hiv_incidence', 'malaria_incidence', 'tb_incidence'])
date_range = st.sidebar.date_input("Date Range", [df['date'].min(), df['date'].max()])

# Filter dataset
filtered_df = df[(df['region'].isin(selected_region)) & 
                 (df['date'] >= pd.to_datetime(date_range[0])) & 
                 (df['date'] <= pd.to_datetime(date_range[1]))]

# Main Title
st.title("📈 Ghana Infectious Disease Trends Dashboard")
st.markdown("#### Machine Learning-Powered Epidemiology | HIV/AIDS Focus")

# Section 1: Time Series Line Chart
st.subheader("1. National Disease Trends Over Time")
fig = px.line(filtered_df, x='date', y=selected_disease, color='region', title=f"{selected_disease.replace('_', ' ').title()} Over Time")
st.plotly_chart(fig, use_container_width=True)

# Section 2: Regional Distribution Map using GeoJSON (10 Regions Corrected)
st.subheader("2. Regional Distribution Map (10 Original Regions)")
latest_df = filtered_df.sort_values('date').groupby('region').tail(1)
latest_df = latest_df.rename(columns={"region": "Region"})

m = folium.Map(location=[7.9, -1.0], zoom_start=6, tiles="CartoDB positron")

choropleth = folium.Choropleth(
    geo_data=geojson_data,
    data=latest_df,
    columns=["Region", selected_disease],
    key_on="feature.properties.shapeName",
    fill_color="YlGnBu",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name=selected_disease.replace("_", " ").title(),
    nan_fill_color="gray"
)
choropleth.add_to(m)
folium.GeoJsonTooltip(fields=["shapeName"]).add_to(choropleth.geojson)
st_folium(m, width=700, height=500)

# Section 3: Behavioral & Demographic Correlation
st.subheader("3. Behavioral & Demographic Correlation")
selected_var = st.selectbox("Choose variable to compare with incidence:", 
                            ['education_access_index', 'condom_use_rate', 'urbanization_level', 
                             'hiv_awareness_index', 'youth_unemployment_rate'])
fig2 = px.scatter(filtered_df, x=selected_var, y=selected_disease, color='region',
                  title=f"{selected_var.replace('_', ' ').title()} vs. {selected_disease.replace('_', ' ').title()}")
st.plotly_chart(fig2, use_container_width=True)

# Section 4: ML Forecasting Results
st.subheader("4. ML Forecasting Results")
st.markdown("🧠 Machine Learning forecasts of HIV/AIDS incidence by region (to 2030)")

# Forecast Line Plot
st.markdown("##### Forecasted HIV Incidence (Selected Regions)")
forecast_df_filtered = forecast_df[forecast_df['region'].isin(selected_region)]
fig3 = px.line(forecast_df_filtered, x="year", y="hiv_predicted", color="region", title="Forecasted HIV Incidence to 2030")
st.plotly_chart(fig3, use_container_width=True)

# Model Performance Table
st.markdown("##### Model Performance Summary")
st.dataframe(metrics_df, use_container_width=True)
