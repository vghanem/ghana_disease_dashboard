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

# Data loading functions
@st.cache_data
def load_main_data():
    df = pd.read_csv("ghana_infectious_disease_model_dataset_cleaned.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['region'] = df['region'].replace(REGION_MAPPING).str.upper()
    return df[df['region'].isin(ORIGINAL_REGIONS)]

@st.cache_data
def load_geojson():
    with open("geoBoundaries-GHA-ADM1_simplified.geojson") as f:
        gj = json.load(f)
        
    valid_features = []
    for feature in gj['features']:
        region_name = feature['properties']['shapeName']
        mapped_name = REGION_MAPPING.get(region_name, region_name).upper()
        if mapped_name in ORIGINAL_REGIONS:
            feature['properties']['shapeName'] = mapped_name
            valid_features.append(feature)
    
    gj['features'] = valid_features
    return gj

@st.cache_data
def load_forecast():
    df = pd.read_csv("hiv_predicted_2030_by_region.csv")
    df['region'] = df['region'].replace(REGION_MAPPING).str.upper()
    return df[df['region'].isin(ORIGINAL_REGIONS)]

@st.cache_data
def load_metrics():
    return pd.read_csv("model_performance_metrics.csv")

# Initialize application
def main():
    # Load datasets
    df = load_main_data()
    geojson = load_geojson()
    forecast_df = load_forecast()
    metrics_df = load_metrics()

    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    
    # Region selector
    all_regions = sorted(df['region'].unique().tolist())
    selected_regions = st.sidebar.multiselect(
        "Select Regions", 
        all_regions, 
        default=all_regions,
        help="Choose regions to analyze"
    )
    
    # Disease selector
    disease_options = ['hiv_incidence', 'malaria_incidence', 'tb_incidence']
    selected_disease = st.sidebar.selectbox(
        "Select Disease Metric", 
        disease_options,
        index=0
    )
    
    # Date controls
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    selected_date = st.sidebar.date_input(
        "Reference Date for Map", 
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

    # Data filtering
    current_data = df[
        (df['region'].isin(selected_regions)) &
        (df['date'].dt.date == selected_date)
    ]
    
    # Main layout
    st.title("🇬🇭 Ghana Infectious Disease Surveillance System")
    st.markdown("### Integrated Disease Modeling & Predictive Analytics Platform")
    st.markdown("---")

    # Section 1: Temporal Trends
    with st.container():
        st.subheader("Temporal Disease Progression")
        trend_df = df[df['region'].isin(selected_regions)]
        fig = px.line(
            trend_df, 
            x='date', 
            y=selected_disease, 
            color='region',
            labels={'date': 'Timeline', selected_disease: 'Cases per 100k'},
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    # Section 2: Geospatial Analysis
    with st.container():
        st.subheader("Regional Disease Distribution")
        
        # Prepare choropleth data
        choropleth_data = current_data.groupby('region')[selected_disease].mean().reset_index()
        choropleth_data.columns = ['Region', 'Value']
        
        # Ensure all regions are represented
        complete_regions = pd.DataFrame({'Region': ORIGINAL_REGIONS})
        choropleth_data = complete_regions.merge(choropleth_data, on='Region', how='left')
        choropleth_data['Value'] = choropleth_data['Value'].fillna(0)
        
        # Create Folium map
        m = folium.Map(location=[7.9465, -1.0232], zoom_start=6.2, tiles='CartoDB positron')
        
        # Create color scale
        max_value = choropleth_data['Value'].max()
        colormap = LinearColormap(
            colors=['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494'],
            vmin=0,
            vmax=max_value if max_value > 0 else 100
        )
        
        # Add choropleth layer
        choropleth = folium.Choropleth(
            geo_data=geojson,
            name='choropleth',
            data=choropleth_data,
            columns=['Region', 'Value'],
            key_on='feature.properties.shapeName',
            fill_color=colormap,
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Cases per 100,000 population',
            highlight=True,
            reset=True
        ).add_to(m)
        
        # Add tooltips
        choropleth.geojson.add_child(
            folium.features.GeoJsonTooltip(
                fields=['shapeName'],
                aliases=['Region: '],
                localize=True
            )
        )
        
        # Display map
        st_folium(m, width=1200, height=600)

    # Section 3: Predictive Analytics
    with st.container():
        st.subheader("HIV Incidence Projections")
        fig_forecast = px.area(
            forecast_df,
            x='year',
            y='hiv_predicted',
            color='region',
            labels={'year': 'Projection Year', 'hiv_predicted': 'Predicted Cases'},
            line_shape='spline'
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

    # Section 4: Correlation Analysis
    with st.container():
        st.subheader("Cross-Disease Correlation Matrix")
        corr_matrix = df[['hiv_incidence', 'malaria_incidence', 'tb_incidence']].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect='auto',
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # Section 5: Model Performance
    with st.container():
        st.subheader("Model Performance Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(
                metrics_df.style.format_index(escape="latex", precision=2),
                use_container_width=True
            )
        
        with col2:
            st.markdown("**Key Metrics Interpretation**")
            st.markdown("""
            - **RMSE**: Root Mean Square Error (lower is better)
            - **MAE**: Mean Absolute Error (lower is better)
            - **R²**: Coefficient of Determination (closer to 1 is better)
            - **MAPE**: Mean Absolute Percentage Error (under 10% is excellent)
            """)

if __name__ == "__main__":
    main()
