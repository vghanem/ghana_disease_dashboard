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

# Original 10 regions (lowercase in original file)
original_regions = [
    'upper west', 'upper east', 'northern', 'brong-ahafo',
    'ashanti', 'eastern', 'western', 'central', 'greater accra', 'volta'
]

# Load main dataset
@st.cache_data
def load_main_data():
    df = pd.read_csv("ghana_infectious_disease_model_dataset_cleaned.csv")
    df['date'] = pd.to_datetime(df['date'])
    # Convert region names to lowercase first, then map and uppercase
    df['region'] = df['region'].str.lower().replace(region_mapping).str.upper()
    df = df[df['region'].isin([r.upper() for r in original_regions])]
    return df

# Load GeoJSON
@st.cache_data
def load_geojson():
    with open("geoBoundaries-GHA-ADM1_simplified.geojson") as f:
        gj = json.load(f)
        # filter features to original regions
        gj['features'] = [feat for feat in gj['features']
                          if feat['properties']['shapeName'].lower() in original_regions]
        # standardize shapeName to uppercase
        for feat in gj['features']:
            feat['properties']['shapeName'] = feat['properties']['shapeName'].upper()
        return gj

# Load forecasts and metrics
@st.cache_data
def load_forecast():
    df = pd.read_csv("hiv_predicted_2030_by_region.csv")
    df['region'] = df['region'].str.lower().replace(region_mapping).str.upper()
    return df

@st.cache_data
def load_metrics():
    # Preprocess the metrics CSV file
    metrics = []
    current_model = None
    with open("model_performance_metrics.csv", "r") as f:
        for line in f:
            if line.strip():  # Check if the line is not empty
                parts = line.strip().split(": ")
                if len(parts) == 2:
                    key, value = parts
                    if key == "Model":
                        current_model = value
                    else:
                        metrics.append({
                            "model": current_model,
                            "metric": key.lower(),
                            "value": value
                        })
    # Convert the list of dictionaries to a DataFrame
    metrics_df = pd.DataFrame(metrics)
    # Check if the DataFrame is not empty before pivoting
    if not metrics_df.empty:
        # Pivot the DataFrame to have models as rows and metrics as columns
        metrics_df = metrics_df.pivot(index='model', columns='metric', values='value')
        # Convert metric values to numeric types
        metrics_df = metrics_df.apply(pd.to_numeric, errors='ignore')
    else:
        # Create an empty DataFrame with the expected structure if no data is found
        metrics_df = pd.DataFrame(columns=['model', 'rmse', 'r2', 'mae', 'mape'])
    return metrics_df.reset_index()

# Data loading
df = load_main_data()
geojson_data = load_geojson()
forecast_df = load_forecast()
metrics_df = load_metrics()

# Sidebar filters
st.sidebar.header("Filter Panel")
# Regions
all_regions = df['region'].unique().tolist()
select_all = st.sidebar.checkbox("Select all regions", True)
selected_region = st.sidebar.multiselect(
    "Regions", all_regions, default=all_regions if select_all else []
)
# Disease
disease_opts = ['hiv_incidence', 'malaria_incidence', 'tb_incidence']
selected_disease = st.sidebar.selectbox("Disease", disease_opts)
# Date range for trend
min_date, max_date = df['date'].min().date(), df['date'].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
# Single date for map/scatter
selected_date = st.sidebar.date_input("Select Date", min_value=min_date, max_value=max_date, value=min_date)

# Slices
df_time = df[(df['region'].isin(selected_region)) &
             (df['date'].dt.date >= date_range[0]) & 
             (df['date'].dt.date <= date_range[1])]
df_single = df[(df['region'].isin(selected_region)) & 
               (df['date'].dt.date == selected_date)]

# Header
st.title("📈 Ghana Infectious Disease Trends Dashboard")
st.markdown("#### Machine Learning-Powered Epidemiology | HIV/AIDS Focus")
st.markdown("---")

# Section 1: Time Series
st.subheader("1. National Disease Trends Over Time")
fig1 = px.line(df_time, x='date', y=selected_disease, color='region')
fig1.update_layout(width=1200, height=600)
st.plotly_chart(fig1, use_container_width=True)

# Section 2: Choropleth Map (10 Original Regions)
st.subheader("2. Regional Distribution Map (10 Original Regions)")

# Prepare latest data for the selected date
latest = df_single.groupby('region').last().reset_index().rename(columns={'region':'Region'})

# Guard against empty data for the map
if latest.empty:
    st.warning("No regional data available for the selected date. Please adjust filters.")
else:
    m = folium.Map(location=[7.9, -1.0], zoom_start=6, tiles="CartoDB positron")
    
    # Create choropleth with proper tooltip integration
    folium.Choropleth(
        geo_data=geojson_data,
        data=latest,
        columns=['Region', selected_disease],
        key_on='feature.properties.shapeName',
        fill_color='YlGnBu',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=selected_disease.replace('_',' ').title(),
        nan_fill_color='gray',
        tooltip=folium.GeoJsonTooltip(
            fields=['shapeName', selected_disease],
            aliases=['Region', selected_disease.replace('_',' ').title()]
        )
    ).add_to(m)
    
    st_folium(m, width=700, height=500)

# Section 3: Behavioral & Demographic Correlation
st.subheader("3. Behavioral & Demographic Correlation")
selected_var = st.selectbox("Choose variable", ['education_access_index','condom_use_rate','urbanization_level','hiv_awareness_index','youth_unemployment_rate'])
fig2 = px.scatter(df_single, x=selected_var, y=selected_disease, color='region')
st.plotly_chart(fig2, use_container_width=True)

# Section 4: Correlation Heatmap (Enlarged)
st.subheader("4. Correlation Heatmap")
num_cols = ['hiv_incidence','malaria_incidence','tb_incidence','education_access_index','condom_use_rate','urbanization_level','hiv_awareness_index','youth_unemployment_rate']
corr = df_time[num_cols].corr()
fig_hm = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix",
                   labels=dict(color="Correlation"), x=corr.columns, y=corr.columns)
fig_hm.update_layout(height=600)  # Enlarged height
st.plotly_chart(fig_hm, use_container_width=True)

# Section 6: Model Performance
st.subheader("6. Model Performance Summary")
st.dataframe(metrics_df, use_container_width=True)

# Section 7: Model Metrics Correlation Heatmap
st.subheader("7. Machine Learning Model Performance Heatmap")

if not metrics_df.empty and 'model' in metrics_df.columns:
    model_metrics = metrics_df.set_index('model')
    fig_mm = px.imshow(model_metrics, text_auto=True, aspect="auto", 
                      title="ML Model Performance Metrics",
                      labels=dict(color="Score"), x=model_metrics.columns, y=model_metrics.index)
    fig_mm.update_layout(height=500)
    st.plotly_chart(fig_mm, use_container_width=True)
else:
    st.warning("No model metrics data available for visualization.")

# Footer
st.markdown("---")
st.markdown("*Developed by Valentine Ghanem | MSc Public Health & Data Science*")
