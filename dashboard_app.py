import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import json
from branca.colormap import LinearColormap

# List of the 10 original regions (uppercase)
original_regions = ['UPPER WEST', 'UPPER EAST', 'NORTHERN', 'BRONG-AHAFO', 'ASHANTI', 'EASTERN', 'WESTERN', 'CENTRAL', 'GREATER ACCRA', 'VOLTA']

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

# Load main dataset
@st.cache_data
def load_main_data():
    df = pd.read_csv("ghana_infectious_disease_model_dataset_cleaned.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['region'] = df['region'].str.lower().replace(REGION_MAPPING).str.upper()
    df = df[df['region'].isin([r.upper() for r in original_regions])]
    return df

# Load GeoJSON
@st.cache_data
def load_geojson():
    with open("geoBoundaries-GHA-ADM1_simplified.geojson") as f:
        gj = json.load(f)
        gj['features'] = [feat for feat in gj['features']
                          if feat['properties']['shapeName'].lower() in original_regions]
        for feat in gj['features']:
            feat['properties']['shapeName'] = feat['properties']['shapeName'].upper()
        return gj

# Load forecasts and metrics
@st.cache_data
def load_forecast():
    df = pd.read_csv("hiv_predicted_2030_by_region.csv")
    df['region'] = df['region'].str.lower().replace(REGION_MAPPING).str.upper()
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
# Regions
all_regions = df['region'].unique().tolist()
select_all_regions = st.sidebar.checkbox("Select all regions", True)
selected_region = st.sidebar.multiselect(
    "Regions", all_regions, default=all_regions if select_all_regions else []
)

# Disease with select all
disease_opts = ['hiv_incidence', 'malaria_incidence', 'tb_incidence']
select_all_diseases = st.sidebar.checkbox("Select all diseases", True)
selected_disease = st.sidebar.multiselect(
    "Disease", disease_opts, default=disease_opts if select_all_diseases else []
)

# Date range
min_date, max_date = df['date'].min().date(), df['date'].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

# Slices
df_time = df[(df['region'].isin(selected_region)) &
             (df['date'].dt.date >= date_range[0]) & 
             (df['date'].dt.date <= date_range[1])]
df_single = df[(df['region'].isin(selected_region)) & 
               (df['date'].dt.date >= date_range[0]) & 
               (df['date'].dt.date <= date_range[1])]

# Get the latest date within the selected range for the map
if not df_single.empty:
    selected_date = df_single['date'].max().date()
    df_single = df_single[df_single['date'].dt.date == selected_date]

# Header
st.title("📈 Ghana Infectious Disease Trends Dashboard")
st.markdown("#### Machine Learning-Powered Epidemiology | HIV/AIDS Focus")
st.markdown("---")

# Section 1: Time Series
st.subheader("1. National Disease Trends Over Time")
if selected_disease:
    fig1 = px.line(df_time, x='date', y=selected_disease, color='region')
    fig1.update_layout(width=1200, height=600)
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.warning("Please select at least one disease to display trends.")

# Section 2: Choropleth Map (10 Original Regions)
st.subheader("2. Regional Distribution Map (10 Original Regions)")

# Prepare latest data for the selected date
if not df_single.empty:
    latest = df_single.groupby('region').last().reset_index().rename(columns={'region':'Region'})

    # Guard against empty data for the map
    if not latest.empty:
        m = folium.Map(location=[7.9, -1.0], zoom_start=6, tiles="Stamen Toner")
        
        # Create choropleth with proper tooltip integration
        folium.Choropleth(
            geo_data=geojson_data,
            data=latest,
            columns=['Region', selected_disease[0] if selected_disease else ''],
            key_on='feature.properties.shapeName',
            fill_color='YlGnBu',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=(selected_disease[0].replace('_',' ').title() if selected_disease else 'Disease Incidence'),
            nan_fill_color='gray',
            tooltip=folium.GeoJsonTooltip(
                fields=['shapeName', (selected_disease[0] if selected_disease else '')],
                aliases=['Region', (selected_disease[0].replace('_',' ').title() if selected_disease else 'Disease Incidence')]
            )
        ).add_to(m)
        
        st_folium(m, width=500, height=900)
    else:
        st.warning("No regional data available for the selected date range. Please adjust filters.")
else:
    st.warning("No data available for the selected date range. Please adjust filters.")

# Section 3: Behavioral & Demographic Correlation
st.subheader("3. Behavioral & Demographic Correlation")
if selected_disease and not df_single.empty:
    selected_var = st.selectbox("Choose variable", ['education_access_index','condom_use_rate','urbanization_level','hiv_awareness_index','youth_unemployment_rate'])
    fig2 = px.scatter(df_single, x=selected_var, y=selected_disease[0], color='region')
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("Please select at least one disease and ensure data is available for the selected date range.")

# Section 4: Correlation Heatmap of Key Predictors
st.subheader("4. Correlation Heatmap of Key Predictors")

# Calculate the correlation matrix
numeric_cols = ['hiv_incidence', 'malaria_incidence', 'tb_incidence', 'education_access_index',
                'condom_use_rate', 'female_literacy_rate', 'youth_unemployment_rate',
                'hiv_awareness_index', 'access_to_art_pct', 'testing_coverage_pct',
                'health_facility_density', 'urbanization_level']
corr_matrix = df[numeric_cols].corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, center=0, annot=True,
            square=True, cbar_kws={"shrink": .8, "label": "Correlation"})

# Customize the plot
plt.title('Correlation Heatmap: Disease Incidences & Socio-Health Indicators', fontsize=16)
plt.tight_layout()

# Display the heatmap in Streamlit
st.pyplot(plt)

# Section 6: Model Performance
st.subheader("6. Model Performance Summary")
st.dataframe(metrics_df, use_container_width=True)

# Section 7: Model Metrics Correlation Heatmap
st.subheader("7. Machine Learning Model Performance Heatmap")

if not metrics_df.empty:
    fig_mm = px.imshow(metrics_df.set_index('Model'), text_auto=True, aspect="auto", 
                      title="ML Model Performance Metrics",
                      labels=dict(color="Score"), x=metrics_df.columns[1:], y=metrics_df['Model'])
    fig_mm.update_layout(height=500)
    st.plotly_chart(fig_mm, use_container_width=True)
else:
    st.warning("No model metrics data available for visualization.")

# Footer
st.markdown("---")
st.markdown("*Developed by Valentine Ghanem | MSc Public Health & Data Science*")
