import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import json
from branca.colormap import LinearColormap

# Region configuration
REGION_MAPPING = {
    'Ahafo': 'Brong-Ahafo',
    'Bono': 'Brong-Ahafo',
    'Bono East': 'Brong-Ahafo',
    'Savannah': 'Northern',
    'North East': 'Northern',
    'Western North': 'Western',
    'Oti': 'Volta'
}

ORIGINAL_REGIONS = [
    'UPPER WEST', 'UPPER EAST', 'NORTHERN', 'BRONG-AHAFO',
    'ASHANTI', 'EASTERN', 'WESTERN', 'CENTRAL', 'GREATER ACCRA', 'VOLTA'
]

# Enhanced data loading with validation
@st.cache_data
def load_main_data():
    df = pd.read_csv("ghana_infectious_disease_model_dataset_cleaned.csv")
    
    # Validate critical columns
    required_columns = ['date', 'region', 'hiv_incidence', 'malaria_incidence', 'tb_incidence']
    if not all(col in df.columns for col in required_columns):
        st.error("Missing required columns in dataset")
        st.stop()
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['region'] = df['region'].replace(REGION_MAPPING).str.upper()
    return df[df['region'].isin(ORIGINAL_REGIONS)]

@st.cache_data
def load_geojson():
    try:
        with open("geoBoundaries-GHA-ADM1_simplified.geojson") as f:
            gj = json.load(f)
        
        valid_features = []
        for feature in gj['features']:
            original_name = feature['properties']['shapeName']
            mapped_name = REGION_MAPPING.get(original_name, original_name).upper()
            if mapped_name in ORIGINAL_REGIONS:
                feature['properties']['shapeName'] = mapped_name
                valid_features.append(feature)
        
        if not valid_features:
            st.error("No valid regions found in GeoJSON")
            st.stop()
        
        gj['features'] = valid_features
        return gj
    except Exception as e:
        st.error(f"Error loading GeoJSON: {str(e)}")
        st.stop()

@st.cache_data
def load_forecast():
    df = pd.read_csv("hiv_predicted_2030_by_region.csv")
    df['region'] = df['region'].replace(REGION_MAPPING).str.upper()
    return df[df['region'].isin(ORIGINAL_REGIONS)]

@st.cache_data
def load_metrics():
    try:
        metrics = []
        with open("model_performance_metrics.csv", "r") as f:
            for line in f:
                if line.strip():  # Check if the line is not empty
                    parts = line.strip().split(": ")
                    if len(parts) == 2:
                        key, value = parts
                        if key == "Model":
                            current_model = value
                            current_entry = {"model": current_model}
                        else:
                            current_entry[key.lower()] = value
                            metrics.append(current_entry)
        metrics_df = pd.DataFrame(metrics)
        
        # Strip any leading/trailing whitespace characters from column names
        metrics_df.columns = metrics_df.columns.str.strip()
        
        # Check if 'model' column exists
        if 'model' not in metrics_df.columns:
            st.error("The 'model' column is missing in the metrics data.")
            st.stop()
        
        # Pivot the DataFrame to have models as rows and metrics as columns
        metrics_df = metrics_df.pivot(index='model', columns='metric', values='value')
        metrics_df = metrics_df.apply(pd.to_numeric, errors='ignore')
        return metrics_df.reset_index()
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")
        st.stop()

def create_choropleth(data, geojson, selected_disease):
    """Create Folium choropleth map with enhanced styling"""
    m = folium.Map(location=[7.9465, -1.0232], zoom_start=6.2, tiles='CartoDB positron')
    
    # Create color scale
    max_value = data['Value'].max()
    colormap = LinearColormap(
        colors=['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494'],
        vmin=0,
        vmax=max_value if max_value > 0 else 100  # Handle zero-case
    )
    
    choropleth = folium.Choropleth(
        geo_data=geojson,
        name='choropleth',
        data=data,
        columns=['Region', 'Value'],
        key_on='feature.properties.shapeName',
        fill_color=colormap,
        fill_opacity=0.7,
        line_opacity=0.4,
        line_weight=0.5,
        legend_name=f'{selected_disease.replace("_", " ").title()} (per 100k)',
        highlight=True,
        reset=True
    ).add_to(m)
    
    # Add tooltips with styling
    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(
            fields=['shapeName'],
            aliases=['Region:'],
            style=(
                "font-family: Arial; font-size: 12px;"
                "background-color: white; border: 1px solid black;"
                "border-radius: 3px; padding: 5px;"
            )
        )
    )
    
    # Add colormap to map
    colormap.caption = 'Incidence Scale'
    colormap.add_to(m)
    
    return m

def main():
    # Load datasets with error handling
    try:
        df = load_main_data()
        geojson = load_geojson()
        forecast_df = load_forecast()
        metrics_df = load_metrics()
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    
    # Region selector with select all
    all_regions = sorted(df['region'].unique().tolist())
    select_all = st.sidebar.checkbox("Select All Regions", value=True)
    selected_regions = st.sidebar.multiselect(
        "Regions", 
        all_regions, 
        default=all_regions if select_all else []
    )
    
    # Disease selector
    selected_disease = st.sidebar.selectbox(
        "Select Disease Metric", 
        ['hiv_incidence', 'malaria_incidence', 'tb_incidence']
    )
    
   
::contentReference[oaicite:0]{index=0}
 
