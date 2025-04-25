import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
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

# Load datasets
@st.cache_data
def load_main_data():
    df = pd.read_csv("ghana_infectious_disease_model_dataset_cleaned.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['region'] = df['region'].replace(region_mapping)
    return df

@st.cache_data
def load_geojson():
    with open("geoBoundaries-GHA-ADM1_simplified.geojson", "r") as f:
        gj = json.load(f)
        # Apply region remapping to GeoJSON
        for feature in gj["features"]:
            name = feature["properties"]["shapeName"]
            if name in region_mapping:
                feature["properties"]["shapeName"] = region_mapping[name]
        return gj

@st.cache_data
def load_forecast():
    df = pd.read_csv("hiv_predicted_2030_by_region.csv")
    df['region'] = df['region'].replace(region_mapping)
    return df

@st.cache_data
def load_metrics():
    return pd.read_csv("model_performance_metrics.csv")

# Load all data
df = load_main_data()
geojson_data = load_geojson()
forecast_df = load_forecast()
metrics_df = load_metrics()

# Sidebar
st.sidebar.header("Filter Panel")

# Select All Regions
all_regions = df['region'].unique().tolist()
select_all_regions = st.sidebar.checkbox("Select all regions", value=True)
if select_all_regions:
    selected_region = st.sidebar.multiselect("Select Region(s):", all_regions, default=all_regions)
else:
    selected_region = st.sidebar.multiselect("Select Region(s):", all_regions)

# Select All Diseases
disease_options = ['hiv_incidence', 'malaria_incidence', 'tb_incidence']
select_all_diseases = st.sidebar.checkbox("Select all diseases", value=True)
if select_all_diseases:
    selected_disease = st.sidebar.selectbox("Disease", disease_options, index=0)
else:
    selected_disease = st.sidebar.selectbox("Disease", disease_options)

date_range = st.sidebar.date_input("Date Range", [df['date'].min(), df['date'].max()])

# Filtered dataset
filtered_df = df[(df['region'].isin(selected_region)) &
                 (df['date'] >= pd.to_datetime(date_range[0])) &
                 (df['date'] <= pd.to_datetime(date_range[1]))]

# Page Title
st.title("📈 Ghana Infectious Disease Trends Dashboard")
st.markdown("#### Machine Learning-Powered Epidemiology | HIV/AIDS Focus")

# Section 1: Trend Chart
st.subheader("1. National Disease Trends Over Time")
fig = px.line(filtered_df, x='date', y=selected_disease, color='region',
              title=f"{selected_disease.replace('_', ' ').title()} Over Time")
st.plotly_chart(fig, use_container_width=True)

# Section 2 & 3: Map and Correlation Side by Side
st.subheader("2. Regional Distribution Map & 3. Behavioral & Demographic Correlation")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Regional Distribution Map (10 Original Regions)**")
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
    st_folium(m, width=350, height=400)

with col2:
    st.markdown("**Behavioral & Demographic Correlation**")
    selected_var = st.selectbox("Choose variable to compare with incidence:",
                                ['education_access_index', 'condom_use_rate', 'urbanization_level',
                                 'hiv_awareness_index', 'youth_unemployment_rate'])
    fig2 = px.scatter(filtered_df, x=selected_var, y=selected_disease, color='region',
                      title=f"{selected_var.replace('_', ' ').title()} vs. {selected_disease.replace('_', ' ').title()}")
    st.plotly_chart(fig2, use_container_width=True)

# Section 4: Forecasts
st.subheader("4. ML Forecasting Results")
st.markdown("🧠 Forecasted HIV Incidence to 2030 using Machine Learning")

# Forecast line chart
st.markdown("##### Forecasted HIV Incidence (Selected Regions)")
forecast_df_filtered = forecast_df[forecast_df['region'].isin(selected_region)]
if not forecast_df_filtered.empty:
    fig3 = px.line(forecast_df_filtered, x="year", y="hiv_predicted", color="region",
                   title="Forecasted HIV Incidence to 2030")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("No forecast data available for selected regions.")

# Performance table
st.markdown("##### Model Performance Summary")
st.dataframe(metrics_df, use
::contentReference[oaicite:0]{index=0}
 
