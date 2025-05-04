import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Ghana Infectious Disease Dashboard",
    layout="wide",  # Force full screen width
    initial_sidebar_state="expanded"
)

# Add custom CSS to minimize spacing
st.markdown(
    """
    <style>
        .stPlotlyChart, .stFolium {
            margin-bottom: 0px !important;
        }
        .subheader, h3, h2, h1 {
            margin-top: 5px !important;
            margin-bottom: 10px !important;
        }
        hr {
            margin-top: 10px !important;
            margin-bottom: 10px !important;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

from PIL import Image

# Load and display the logo in the sidebar
logo = Image.open("ghana_health_logo.png")
st.sidebar.image(logo, use_container_width=True)

import pandas as pd
import geopandas as gpd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import json

# --- DATA LOADING FUNCTIONS ---
@st.cache_data
def load_main_data():
    df = pd.read_csv("ghana_infectious_disease_model_dataset_cleaned.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['region'] = df['region'].str.upper()
    return df

@st.cache_data
def load_geojson():
    with open("GHA_10regions_merged_final.geojson") as f:
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

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Panel")
select_all_regions = st.sidebar.checkbox("Select all regions", True)
selected_regions = st.sidebar.multiselect("Regions", original_regions, default=original_regions if select_all_regions else [])

disease_opts = ['hiv_incidence', 'malaria_incidence', 'tb_incidence']
select_all_diseases = st.sidebar.checkbox("Select all diseases", True)
selected_diseases = st.sidebar.multiselect("Disease", disease_opts, default=disease_opts if select_all_diseases else [])

# --- DATE RANGE FILTER ---
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
from PIL import Image
import streamlit as st

# Load the Ghana Health logo
logo = Image.open("ghana_health_logo.png")  # Ensure this file exists in your app directory

# Layout: two columns, one for logo, one for text
col1, col2 = st.columns([1, 10])

with col1:
    st.image(logo, width=50)  # Slightly smaller for better alignment

with col2:
    st.markdown(
        """
        <style>
        .dashboard-header {
            display: flex;
            flex-direction: column;
            justify-content: center;
            line-height: 1.2;
            margin-bottom: 10px;
            margin-top: -10px;
        }
        .dashboard-header h1 {
            margin: 0;
            font-size: 30px;
            color: #CE1126;
        }
        .dashboard-header h4 {
            margin: 0;
            font-size: 16px;
            color: #FFD700;
        }
        .dashboard-header span {
            color: #21BF73;
        }
        </style>
        <div class='dashboard-header'>
            <h1>Ghana Infectious Disease Trends Dashboard</h1>
            <h4>Machine Learning-Powered Epidemiology | <span>HIV/AIDS Focus</span></h4>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- SECTION 2: Regional Distribution Map (10 Original Regions) ---
st.subheader("2. Regional Distribution Map (10 Original Regions)")

with st.container():
    st.markdown("### Select Disease to Display on the Map")

    map_disease_option = st.selectbox(
        "Choose disease prevalence for map shading (affects only map):",
        options=['hiv_incidence', 'malaria_incidence', 'tb_incidence'],
        index=0,
        key="map_disease_option"
    )

    # Optional: Inject map styling fix
    st.markdown("""
    <style>
        iframe[title="streamlit_folium.st_folium"] {
            height: 500px !important;
            width: 100% !important;
            display: block;
            margin: 0 auto;
        }
    </style>
    """, unsafe_allow_html=True)

    try:
        latest = df.copy()
        latest['region'] = latest['region'].str.strip().str.title()

        gdf = gpd.read_file("GHA_10regions_merged_final.geojson")
        gdf['shapeName'] = gdf['shapeName'].str.replace(' Region', '', case=False).str.strip().str.title()
        gdf['shapeName'] = gdf['shapeName'].replace({'Brong Ahafo': 'Brong-Ahafo'})

        latest_filtered = latest.sort_values('date').groupby('region').last().reset_index()

        merged = gdf.merge(
            latest_filtered, how='left',
            left_on='shapeName',
            right_on='region'
        )

        if 'date' in merged.columns:
            merged = merged.drop(columns=['date'])

        # Color scale
        color_scale = {
            'hiv_incidence': 'Reds',
            'malaria_incidence': 'Greens',
            'tb_incidence': 'Blues'
        }.get(map_disease_option, 'YlOrRd')

        m = folium.Map(location=[7.9465, -1.0232], zoom_start=6, tiles="CartoDB positron")

        folium.Choropleth(
            geo_data=json.loads(merged.to_json()),
            data=merged,
            columns=['shapeName', map_disease_option],
            key_on='feature.properties.shapeName',
            fill_color=color_scale,
            fill_opacity=0.8,
            line_opacity=0.2,
            nan_fill_color='white',
            legend_name=f"{map_disease_option.replace('_', ' ').title()} per 100k",
            highlight=True,
            line_color='black'
        ).add_to(m)

        folium.GeoJson(
            merged,
            style_function=lambda x: {'fillColor': '#ffffff', 'color': 'black', 'fillOpacity': 0, 'weight': 1},
            highlight_function=lambda x: {'fillColor': '#000000', 'color': '#000000', 'fillOpacity': 0.5, 'weight': 3},
            tooltip=folium.features.GeoJsonTooltip(
                fields=['shapeName', map_disease_option],
                aliases=['Region:', f'{map_disease_option.replace("_", " ").title()}:'],
                localize=True,
                sticky=True
            )
        ).add_to(m)

        st_folium(m, use_container_width=True)

    except Exception as e:
        st.error(f"Map error: {e}")

# --- SECTION 3: Behavioral & Demographic Correlation ---
st.markdown("<h3 style='margin-top: 5px; margin-bottom: 10px;'>3. Behavioral & Demographic Correlation</h3>", unsafe_allow_html=True)

if selected_diseases and not df_single.empty:
    selected_var = st.selectbox(
        "Choose a behavioral or demographic variable:",
        [
            'education_access_index',
            'condom_use_rate',
            'urbanization_level',
            'hiv_awareness_index',
            'youth_unemployment_rate'
        ]
    )

    fig2 = px.scatter(
        df_single,
        x=selected_var,
        y=selected_diseases[0],
        color='region',
        title=f"{selected_diseases[0].replace('_', ' ').title()} vs {selected_var.replace('_', ' ').title()}"
    )

    fig2.update_layout(
        font=dict(family="Arial", size=14, color="white"),
        legend=dict(orientation="v", bgcolor="rgba(0,0,0,0.5)", font=dict(size=12)),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117"
    )

    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("Please select a disease and ensure data is available for the selected date.")
# --- SECTION 4: Correlation Heatmap ---
st.subheader("4. Correlation Heatmap of Key Predictors")
numeric_cols = ['hiv_incidence', 'malaria_incidence', 'tb_incidence', 
               'education_access_index', 'condom_use_rate', 
               'female_literacy_rate', 'youth_unemployment_rate',
               'hiv_awareness_index', 'access_to_art_pct', 
               'testing_coverage_pct', 'health_facility_density', 
               'urbanization_level']
corr = df[numeric_cols].corr()
fig = px.imshow(corr, text_auto=True, aspect='auto', color_continuous_scale='RdBu_r',
                range_color=(-1, 1), labels=dict(color="Correlation"),
                title="Correlation Heatmap: Health Indicators & Disease Incidence")
fig.update_layout(
    width=1400,
    height=700,
    xaxis_title="Variables",
    yaxis_title="Variables",
    coloraxis_colorbar=dict(title="Correlation", thickness=25, len=0.75, yanchor="top", y=0.9),
    font=dict(family="Arial", size=14, color="white"),
    plot_bgcolor="#0E1117",
    paper_bgcolor="#0E1117"
)

fig.update_xaxes(tickangle=45)
fig.update_traces(hoverongaps=False)
st.plotly_chart(fig, use_container_width=True)
st.markdown("""<hr style='margin: 30px 0;'>""", unsafe_allow_html=True)

# --- SECTION 5: Forecasts ---
st.subheader("5. Disease Incidence Forecasts (2030)")
if not forecast_df.empty:
    forecast_cols = forecast_df.columns.tolist()
    y_col = [col for col in forecast_cols if 'predict' in col.lower()]
    if y_col:
        fig5 = px.bar(forecast_df, x='region', y=y_col[0], color='region',
                     barmode='group', title='Projected 2030 Disease Incidence by Region')
        fig5.update_layout(
    xaxis_title='Region',
    yaxis_title='Predicted Incidence Rate',
    xaxis_tickangle=-45,
    font=dict(family="Arial", size=14, color="white"),
    legend=dict(orientation="v", bgcolor="rgba(0,0,0,0.5)", font=dict(size=12)),
    plot_bgcolor="#0E1117",
    paper_bgcolor="#0E1117"
)

        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.warning("No predicted incidence column found in forecast dataset.")
else:
    st.warning("Forecast data not available.")
st.markdown("""<hr style='margin: 30px 0;'>""", unsafe_allow_html=True)

# --- SECTION 6: Model Performance Table ---
st.subheader("6. Model Performance Summary")
st.dataframe(metrics_df, use_container_width=True)
st.markdown("""<hr style='margin: 30px 0;'>""", unsafe_allow_html=True)

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
            yaxis=dict(autorange="reversed"),
            font=dict(family="Arial", size=14, color="white"),
            plot_bgcolor="#0E1117",
            paper_bgcolor="#0E1117"
        )

        st.plotly_chart(fig_perf, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to plot model performance heatmap: {e}")
else:
    st.warning("Model performance data not available.")


# --- SECTION 8: Granular HIV Trends by Region Over Time ---
st.subheader("8. Granular HIV Trends by Region Over Time")
try:
    hiv_heatmap_data = df[['date', 'region', 'hiv_incidence']]
    hiv_heatmap_data = hiv_heatmap_data.groupby(['region', 'date'])['hiv_incidence'].mean().reset_index()
    heatmap_pivot = hiv_heatmap_data.pivot(index='region', columns='date', values='hiv_incidence')
    fig_hiv = px.imshow(heatmap_pivot, labels=dict(x="Date", y="Region", color="HIV Incidence"),
                        aspect='auto', color_continuous_scale='Viridis',
                        title="Granular View: Monthly HIV Incidence by Region (1970–2020)")
    fig_hiv.update_layout(
    width=1000,
    height=700,
    xaxis=dict(tickangle=-45, nticks=25),
    yaxis=dict(autorange="reversed"),
    font=dict(family="Arial", size=14, color="white"),
    plot_bgcolor="#0E1117",
    paper_bgcolor="#0E1117"
)

    st.plotly_chart(fig_hiv, use_container_width=True)
except Exception as e:
    st.error(f"Failed to generate HIV heatmap: {e}")

# --- FOOTER ---
st.markdown("""
---
<p style="text-align: center; font-size: 14px; color: grey;">
Developed by <strong>Valentine Ghanem</strong> | 🇬🇭 <br>
<a href="https://www.valentineghanem.com" target="_blank" style="color:#F63366;">Website</a> |
<a href="https://www.linkedin.com/in/valentineghanem/" target="_blank" style="color:#F63366;">LinkedIn</a> |
<a href="https://doi.org/10.5281/zenodo.15292209" target="_blank" style="color:#F63366;">DOI: 10.5281/zenodo.15292209</a>
</p>
<p style="text-align: center;">
<a href="https://doi.org/10.5281/zenodo.15292209" target="_blank">
<img src="https://zenodo.org/badge/DOI/10.5281/zenodo.15292209.svg" alt="DOI Badge">
</a>
</p>
""", unsafe_allow_html=True)
