
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("ghana_infectious_disease_model_dataset_cleaned.csv")

df = load_data()
df['date'] = pd.to_datetime(df['date'])

# Sidebar - Filter controls
st.sidebar.header("Filter Panel")
selected_region = st.sidebar.multiselect("Select Region(s):", df['region'].unique(), default=df['region'].unique())
selected_disease = st.sidebar.selectbox("Disease", ['hiv_incidence', 'malaria_incidence', 'tb_incidence'])
date_range = st.sidebar.date_input("Date Range", [df['date'].min(), df['date'].max()])

# Filter data
filtered_df = df[(df['region'].isin(selected_region)) & 
                 (df['date'] >= pd.to_datetime(date_range[0])) & 
                 (df['date'] <= pd.to_datetime(date_range[1]))]

# Main page
st.title("📈 Ghana Infectious Disease Trends Dashboard")
st.markdown("#### Machine Learning-Powered Epidemiology | HIV/AIDS Focus")

st.subheader("1. National Disease Trends Over Time")
fig = px.line(filtered_df, x='date', y=selected_disease, color='region', title=f"{selected_disease.replace('_', ' ').title()} Over Time")
st.plotly_chart(fig, use_container_width=True)

import json

st.subheader("2. Regional Distribution Map (GeoJSON Choropleth)")

# Load geoJSON
with open("geoBoundaries-GHA-ADM1_simplified.geojson", "r") as f:
    geojson_data = json.load(f)

# Prepare latest value per region for selected disease
latest_df = filtered_df.sort_values('date').groupby('region').tail(1)

# Rename for consistency (if necessary)
latest_df = latest_df.rename(columns={"region": "Region"})

# Create Folium Map
m = folium.Map(location=[7.9, -1.0], zoom_start=6, tiles="CartoDB positron")

# Choropleth layer
choropleth = folium.Choropleth(
    geo_data=geojson_data,
    data=latest_df,
    columns=["Region", selected_disease],
    key_on="feature.properties.shapeName",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name=selected_disease.replace("_", " ").title(),
    nan_fill_color="gray"
)
choropleth.add_to(m)

# Add interactivity with tooltips
folium.GeoJsonTooltip(fields=["shapeName"]).add_to(choropleth.geojson)

# Render the map in Streamlit
st_folium(m, width=700, height=500)


st.subheader("3. Behavioral & Demographic Correlation")
selected_var = st.selectbox("Choose variable to compare with incidence:", 
                            ['education_access_index', 'condom_use_rate', 'urbanization_level', 
                             'hiv_awareness_index', 'youth_unemployment_rate'])

fig2 = px.scatter(filtered_df, x=selected_var, y=selected_disease, color='region',
                  title=f"{selected_var.replace('_', ' ').title()} vs. {selected_disease.replace('_', ' ').title()}")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("4. ML Forecasting Results")
st.markdown("🧠 Coming Soon: Machine learning forecasts of HIV/AIDS incidence to 2030.")

# Optional: If CSV of forecasts available
# forecast_df = pd.read_csv("hiv_forecast.csv")
# fig3 = px.line(forecast_df, x="date", y="predicted", title="Forecasted HIV Incidence")
# st.plotly_chart(fig3)
