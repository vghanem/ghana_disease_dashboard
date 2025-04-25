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

# Original 10 regions
original_regions = [
    'UPPER WEST', 'UPPER EAST', 'NORTHERN', 'BRONG-AHAFO',
    'ASHANTI', 'EASTERN', 'WESTERN', 'CENTRAL', 'GREATER ACCRA', 'VOLTA'
]

# Load main dataset
@st.cache_data
def load_main_data():
    df = pd.read_csv("ghana_infectious_disease_model_dataset_cleaned.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['region'] = df['region'].replace(region_mapping).str.upper()
    df = df[df['region'].isin(original_regions)]
    return df

# Load GeoJSON
@st.cache_data
def load_geojson():
    with open("geoBoundaries-GHA-ADM1_simplified.geojson") as f:
        gj = json.load(f)
        # Filter features to original regions using correct property name
        gj['features'] = [feat for feat in gj['features']
                          if feat['properties'].get('NAME_1', '').upper() in original_regions]
        # Standardize region names
        for feat in gj['features']:
            name = feat['properties'].get('NAME_1', '')  # Use actual property name from your GeoJSON
            feat['properties']['NAME_1'] = region_mapping.get(name, name).upper()
        return gj

# Load forecasts and metrics
@st.cache_data
def load_forecast():
    df = pd.read_csv("hiv_predicted_2030_by_region.csv")
    df['region'] = df['region'].replace(region_mapping).str.upper()
    return df

@st.cache_data
def load_metrics():
    return pd.read_csv("model_performance_metrics.csv")

# Data loading
df = load_main_data()
geojson_data = load_geojson()
forecast_df = load_forecast()
metrics_df = load_metrics()

# Sidebar filters
st.sidebar.header("Filter Panel")
select_all = st.sidebar.checkbox("Select all regions", True)
selected_region = st.sidebar.multiselect(
    "Regions", 
    df['region'].unique().tolist(), 
    default=df['region'].unique().tolist() if select_all else []
)
selected_disease = st.sidebar.selectbox("Disease", ['hiv_incidence', 'malaria_incidence', 'tb_incidence'])
date_range = st.sidebar.date_input("Date Range", [df['date'].min().date(), df['date'].max().date()])
selected_date = st.sidebar.date_input("Select Date", min_value=df['date'].min().date())

# Data slices
df_time = df[(df['region'].isin(selected_region)) &
             (df['date'].dt.date.between(date_range[0], date_range[1]))]
df_single = df[(df['region'].isin(selected_region)) &
               (df['date'].dt.date == selected_date)]

# Header
st.title("📈 Ghana Infectious Disease Trends Dashboard")
st.markdown("#### Machine Learning-Powered Epidemiology | HIV/AIDS Focus")
st.markdown("---")

# Section 2: Choropleth Map (Fixed)
st.subheader("2. Regional Distribution Map (10 Original Regions)")

if df_single.empty:
    st.warning("No data available for selected date. Please adjust filters.")
else:
    # Create map
    m = folium.Map(location=[7.9, -1.0], zoom_start=6, tiles="CartoDB positron")
    
    # Add Choropleth layer
    choropleth = folium.Choropleth(
        geo_data=geojson_data,
        data=df_single.groupby('region').last().reset_index(),
        columns=['region', selected_disease],
        key_on='feature.properties.NAME_1',  # Updated to match GeoJSON property
        fill_color='YlGnBu',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=selected_disease.replace('_',' ').title(),
        nan_fill_color='gray',
        highlight=True
    ).add_to(m)
    
    # Add tooltips
    folium.GeoJson(
        geojson_data,
        name='Regions',
        tooltip=folium.GeoJsonTooltip(
            fields=['NAME_1'],  # Match GeoJSON property
            aliases=['Region:']
        ),
        style_function=lambda x: {'fillOpacity': 0, 'stroke': False}
    ).add_to(m)
    
    # Render map
    st_folium(m, width=700, height=500, returned_objects=[])

# Other sections remain the same...
# (Keep your existing code for sections 1, 3-7 and the footer)
