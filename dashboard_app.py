import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import json

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

# --- SECTION 2: Corrected Interactive Choropleth Map ---
st.subheader("2. Regional Distribution Map (10 Original Regions)")
if not df_single.empty and selected_diseases:
    latest = df_single.groupby('region').last().reset_index()

    try:
        # Load GeoJSON as GeoDataFrame
        gdf = gpd.read_file("geoBoundaries-GHA-ADM1_simplified.geojson")
        gdf['shapeName'] = gdf['shapeName'].str.upper()

        # Merge data
        merged = gdf.set_index('shapeName').join(
            latest.set_index('region')
        ).reset_index()

        m = folium.Map(location=[7.9465, -1.0232], zoom_start=6, tiles="CartoDB positron")

        folium.Choropleth(
            geo_data=merged,
            data=merged,
            columns=['shapeName', selected_diseases[0]],
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
            merged,
            name="Regions",
            style_function=lambda feature: {
                "fillOpacity": 0,
                "color": "black",
                "weight": 1,
                "dashArray": "5, 5"
            },
            tooltip=folium.features.GeoJsonTooltip(
                fields=['shapeName', selected_diseases[0]],
                aliases=['Region:', f'{selected_diseases[0].replace("_", " ").title()}:'],
                localize=True
            )
        ).add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(m, width=900, height=650)

    except Exception as e:
        st.error(f"Map error: {e}")
else:
    st.warning("Select a disease and ensure data is available.")

# --- SECTION 3: Behavioral & Demographic Correlation ---
st.subheader("3. Behavioral & Demographic Correlation")
if selected_diseases and not df_single.empty:
    selected_var = st.selectbox("Choose variable", 
                               ['education_access_index','condom_use_rate',
                                'urbanization_level','hiv_awareness_index',
                                'youth_unemployment_rate'])
    fig2 = px.scatter(df_single, x=selected_var, y=selected_diseases[0], color='region')
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("Select a disease and ensure data is available.")

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
fig.update_layout(width=800, height=700, xaxis_title="Variables", yaxis_title="Variables",
                  coloraxis_colorbar=dict(title="Correlation", thickness=25, len=0.75, yanchor="top", y=0.9))
fig.update_xaxes(tickangle=45)
fig.update_traces(hoverongaps=False)
st.plotly_chart(fig, use_container_width=True)

# --- SECTION 5: Forecasts ---
st.subheader("5. Disease Incidence Forecasts (2030)")
if not forecast_df.empty:
    forecast_cols = forecast_df.columns.tolist()
    y_col = [col for col in forecast_cols if 'predict' in col.lower()]
    if y_col:
        fig5 = px.bar(forecast_df, x='region', y=y_col[0], color='region',
                     barmode='group', title='Projected 2030 Disease Incidence by Region')
        fig5.update_layout(xaxis_title='Region', yaxis_title='Predicted Incidence Rate', xaxis_tickangle=-45)
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.warning("No predicted incidence column found in forecast dataset.")
else:
    st.warning("Forecast data not available.")

# --- SECTION 6: Model Performance Table ---
st.subheader("6. Model Performance Summary")
st.dataframe(metrics_df, use_container_width=True)

# --- SECTION 7: Interactive Model Performance Heatmap (Direct from Section 6 Table) ---
st.subheader("7. Interactive Model Performance Heatmap")

if not metrics_df.empty:
    try:
        # Directly use the metrics_df (assuming it's already in wide format)
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


# --- SECTION 8: Granular HIV Trends by Region Over Time ---
st.subheader("8. Granular HIV Trends by Region Over Time")
try:
    hiv_heatmap_data = df[['date', 'region', 'hiv_incidence']]
    hiv_heatmap_data = hiv_heatmap_data.groupby(['region', 'date'])['hiv_incidence'].mean().reset_index()
    heatmap_pivot = hiv_heatmap_data.pivot(index='region', columns='date', values='hiv_incidence')
    fig_hiv = px.imshow(heatmap_pivot, labels=dict(x="Date", y="Region", color="HIV Incidence"),
                        aspect='auto', color_continuous_scale='Viridis',
                        title="Granular View: Monthly HIV Incidence by Region (1970–2020)")
    fig_hiv.update_layout(width=1000, height=700, xaxis=dict(tickangle=-45, nticks=25), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_hiv, use_container_width=True)
except Exception as e:
    st.error(f"Failed to generate HIV heatmap: {e}")

# --- FOOTER ---
st.markdown("---")
st.markdown("🌐 *Developed by Valentine Ghanem | MSc Public Health & Data Science*")
