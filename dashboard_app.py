import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import json

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Ghana Disease Trends Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- DATA LOADING FUNCTIONS ---
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
        features = [feat for feat in gj['features'] if feat['properties']['shapeName'].upper() in original_regions]
        gj['features'] = features
        return gj

@st.cache_data
def load_forecast():
    df = pd.read_csv("hiv_predicted_2030_by_region.csv")
    df['region'] = df['region'].str.upper()
    return df

@st.cache_data
def load_metrics():
    return pd.read_csv("model_performance_metrics.csv")

# --- LOAD DATA ---
df = load_main_data()
original_regions = df['region'].unique().tolist()
geojson_data = load_geojson()
forecast_df = load_forecast()
metrics_df = load_metrics()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Panel")
select_all_regions = st.sidebar.checkbox("Select all regions", True)
selected_regions = st.sidebar.multiselect("Regions", original_regions, default=original_regions if select_all_regions else [])

disease_opts = ['hiv_incidence', 'malaria_incidence', 'tb_incidence']
select_all_diseases = st.sidebar.checkbox("Select all diseases", True)
selected_diseases = st.sidebar.multiselect("Disease", disease_opts, default=disease_opts if select_all_diseases else [])

min_date = df['date'].min().date()
max_date = df['date'].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

# --- FILTER DATA ---
df_time = df[(df['region'].isin(selected_regions)) & 
             (df['date'].dt.date >= date_range[0]) & 
             (df['date'].dt.date <= date_range[1])]

df_single = df_time.copy()
if not df_single.empty:
    selected_date = df_single['date'].max().date()
    df_single = df_single[df_single['date'].dt.date == selected_date]

# --- HEADER ---
st.title("📈 Ghana Infectious Disease Trends Dashboard")
st.markdown("#### Machine Learning-Powered Epidemiology | HIV/AIDS Focus")
st.markdown("---")

# --- SECTION 1: Time Series ---
st.subheader("1. National Disease Trends Over Time")
if not selected_diseases:
    st.warning("Please select at least one disease to display trends.")
elif df_time.empty:
    st.warning("No data available for selected filters.")
else:
    fig1 = px.line(df_time, x='date', y=selected_diseases, color='region')
    fig1.update_layout(width=1200, height=600, xaxis=dict(tickangle=-45))
    st.plotly_chart(fig1, use_container_width=True)

    # --- DOWNLOAD FILTERED DATA ---
    csv = df_time.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Filtered Data", csv, "filtered_disease_data.csv", "text/csv")

# --- SECTION 2: Interactive Choropleth Map (Filtered to 10 Original Regions) ---
st.subheader("2. Regional Distribution Map (10 Original Regions)")
if not df_single.empty and selected_diseases:
    latest = df_single.groupby('region').last().reset_index()
    try:
        m = folium.Map(location=[7.9465, -1.0232], zoom_start=6, tiles="CartoDB positron")

        folium.Choropleth(
            geo_data=geojson_data,
            data=latest,
            columns=['region', selected_diseases[0]],
            key_on='feature.properties.shapeName',
            fill_color='YlOrRd',
            fill_opacity=0.8,
            line_opacity=0.2,
            nan_fill_color='white',
            legend_name=f"{selected_diseases[0].replace('_', ' ').title()} per 100k",
            highlight=True,
            line_color='black'
        ).add_to(m)

        folium.GeoJson(
            geojson_data,
            name="Regions",
            style_function=lambda feature: {
                "fillOpacity": 0,
                "color": "black",
                "weight": 1,
                "dashArray": "5, 5"
            },
            tooltip=folium.features.GeoJsonTooltip(
                fields=['shapeName'],
                aliases=['Region:'],
                localize=True
            )
        ).add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(m, width=900, height=650)

    except Exception as e:
        st.error(f"Map error: {e}")
else:
    st.warning("Select a disease and ensure data is available.")

# --- SECTION 6: Model Performance Table ---
st.subheader("6. Model Performance Summary")
st.dataframe(metrics_df, use_container_width=True)

# --- SECTION 7: Interactive Model Performance Heatmap ---
st.subheader("7. Interactive Model Performance Heatmap")
if not metrics_df.empty:
    try:
        pivot_df = metrics_df.set_index('Model')

        fig_perf = px.imshow(
            pivot_df,
            text_auto=".2f",
            color_continuous_scale='RdBu',
            aspect='auto',
            title="Model Performance Heatmap"
        )

        fig_perf.update_layout(
            width=800,
            height=600,
            xaxis_title="Metrics",
            yaxis_title="Models",
            coloraxis_colorbar=dict(title="Score"),
            xaxis=dict(side="top", tickangle=45),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to plot model performance heatmap: {e}")
else:
    st.warning("Model performance data not available.")
