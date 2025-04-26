import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import json
from branca.colormap import LinearColormap

# --- DATA LOADING ---
@st.cache_data
def load_main_data():
    df = pd.read_csv("ghana_disease_data_10regions.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['region'] = df['region'].str.upper()
    return df

@st.cache_data
def load_geojson():
    with open("geoBoundaries-GHA-ADM1_simplified.geojson") as f:
        gj = json.load(f)
        for feature in gj['features']:
            feature['properties']['shapeName'] = feature['properties']['shapeName'].upper()
        return gj

@st.cache_data
def load_forecast():
    df = pd.read_csv("hiv_predicted_2030_by_region.csv")
    df['region'] = df['region'].str.upper()
    return df

@st.cache_data
def load_metrics():
    return pd.read_csv("model_performance_metrics.csv")

# --- HELPER FUNCTION ---
def get_region_centroid(region_geojson):
    coords = region_geojson['geometry']['coordinates'][0]
    lats = [coord[1] for coord in coords]
    lons = [coord[0] for coord in coords]
    return [sum(lats)/len(lats), sum(lons)/len(lons)]

# --- LOAD DATA ---
df = load_main_data()
geojson_data = load_geojson()
forecast_df = load_forecast()
metrics_df = load_metrics()

original_regions = df['region'].unique().tolist()

# --- SIDEBAR ---
st.sidebar.header("Filter Panel")
select_all_regions = st.sidebar.checkbox("Select all regions", True)
selected_regions = st.sidebar.multiselect("Regions", original_regions, default=original_regions if select_all_regions else [])

disease_opts = ['hiv_incidence', 'malaria_incidence', 'tb_incidence']
select_all_diseases = st.sidebar.checkbox("Select all diseases", True)
selected_diseases = st.sidebar.multiselect("Disease", disease_opts, default=disease_opts if select_all_diseases else [])

min_date,_
