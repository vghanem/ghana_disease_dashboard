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
     gdf = gpd.read_file("geoBoundaries-GHA-ADM1_simplified.geojson")
     gdf['shapeName'] = gdf['shapeName'].str.upper()
     return gdf
     with open("geoBoundaries-GHA-ADM1_simplified.geojson") as f:
         gj = json.load(f)
         features = [feat for feat in gj['features'] if feat['properties']['shapeName'].upper() in original_regions]
         gj['features'] = features
         return gj
 
 @st.cache_data
 def load_forecast():
 @@ -39,12 +41,11 @@
 
 # --- LOAD DATA ---
 df = load_main_data()
 original_regions = df['region'].unique().tolist()
 geojson_data = load_geojson()
 forecast_df = load_forecast()
 metrics_df = load_metrics()
 
 original_regions = df['region'].unique().tolist()
 
 # --- SIDEBAR FILTERS ---
 st.sidebar.header("Filter Panel")
 select_all_regions = st.sidebar.checkbox("Select all regions", True)
 @@ -93,18 +94,12 @@
 if not df_single.empty and selected_diseases:
     latest = df_single.groupby('region').last().reset_index()
     try:
         gdf = geojson_data.copy()
 
         gdf_10 = gdf[gdf['shapeName'].isin(latest['region'].unique())].copy()
         merged = gdf_10.set_index('shapeName').join(latest.set_index('region')).reset_index()
 
         m = folium.Map(location=[7.9465, -1.0232], zoom_start=6, tiles="CartoDB positron")
 
         folium.Choropleth(
             geo_data=merged.to_json(),
             name='choropleth',
             data=merged,
             columns=['shapeName', selected_diseases[0]],
             geo_data=geojson_data,
             data=latest,
             columns=['region', selected_diseases[0]],
             key_on='feature.properties.shapeName',
             fill_color='YlOrRd',
             fill_opacity=0.8,
 @@ -116,7 +111,7 @@
         ).add_to(m)
 
         folium.GeoJson(
             merged.to_json(),
             geojson_data,
             name="Regions",
             style_function=lambda feature: {
                 "fillOpacity": 0,
 @@ -125,8 +120,8 @@
                 "dashArray": "5, 5"
             },
             tooltip=folium.features.GeoJsonTooltip(
                 fields=['shapeName', selected_diseases[0]],
                 aliases=['Region:', f'{selected_diseases[0].replace("_", " ").title()}:'],
                 fields=['shapeName'],
                 aliases=['Region:'],
                 localize=True
             )
         ).add_to(m)
 @@ -139,79 +134,36 @@
 else:
     st.warning("Select a disease and ensure data is available.")
 
 # --- SECTION 3: Behavioral & Demographic Correlation ---
 st.subheader("3. Behavioral & Demographic Correlation")
 if selected_diseases and not df_single.empty:
     selected_var = st.selectbox("Choose variable", 
         ['education_access_index','condom_use_rate','urbanization_level','hiv_awareness_index','youth_unemployment_rate'])
     fig2 = px.scatter(df_single, x=selected_var, y=selected_diseases[0], color='region')
     st.plotly_chart(fig2, use_container_width=True)
 else:
     st.warning("Select a disease and ensure data is available.")
 
 # --- SECTION 4: Correlation Heatmap ---
 st.subheader("4. Correlation Heatmap of Key Predictors")
 numeric_cols = ['hiv_incidence', 'malaria_incidence', 'tb_incidence', 'education_access_index', 'condom_use_rate', 
                 'female_literacy_rate', 'youth_unemployment_rate', 'hiv_awareness_index', 'access_to_art_pct', 
                 'testing_coverage_pct', 'health_facility_density', 'urbanization_level']
 corr = df[numeric_cols].corr()
 fig = px.imshow(corr, text_auto=True, aspect='auto', color_continuous_scale='RdBu_r', range_color=(-1, 1), 
                 labels=dict(color="Correlation"), title="Correlation Heatmap: Health Indicators & Disease Incidence")
 fig.update_layout(width=800, height=700, xaxis_title="Variables", yaxis_title="Variables",
                   coloraxis_colorbar=dict(title="Correlation", thickness=25, len=0.75, yanchor="top", y=0.9))
 fig.update_xaxes(tickangle=45)
 st.plotly_chart(fig, use_container_width=True)
 
 # --- SECTION 5: Forecasts ---
 st.subheader("5. Disease Incidence Forecasts (2030)")
 if not forecast_df.empty:
     y_col = [col for col in forecast_df.columns if 'predict' in col.lower()]
     if y_col:
         fig5 = px.bar(forecast_df, x='region', y=y_col[0], color='region', barmode='group', 
                      title='Projected 2030 Disease Incidence by Region')
         fig5.update_layout(xaxis_title='Region', yaxis_title='Predicted Incidence Rate', xaxis_tickangle=-45)
         st.plotly_chart(fig5, use_container_width=True)
     else:
         st.warning("No predicted incidence column found in forecast dataset.")
 else:
     st.warning("Forecast data not available.")
 
 # --- SECTION 6: Model Performance Table ---
 st.subheader("6. Model Performance Summary")
 st.dataframe(metrics_df, use_container_width=True)
 
 # --- SECTION 7: Model Performance Heatmap ---
 # --- SECTION 7: Interactive Model Performance Heatmap ---
 st.subheader("7. Interactive Model Performance Heatmap")
 if not metrics_df.empty:
     try:
         metrics_long = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
         pivot_df = metrics_long.pivot(index="Model", columns="Metric", values="Value")
 
         fig_perf = px.imshow(pivot_df, text_auto=".2f", color_continuous_scale='RdBu', aspect='auto', 
                              title="Model Performance Heatmap")
         fig_perf.update_layout(width=800, height=600, xaxis_title="Metrics", yaxis_title="Models",
                                coloraxis_colorbar=dict(title="Score"), xaxis=dict(side="top", tickangle=45), 
                                yaxis=dict(autorange="reversed"))
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
         st.error(f"Failed to pivot and plot model performance: {e}")
         st.error(f"Failed to plot model performance heatmap: {e}")
 else:
     st.warning("Model performance data not available.")
 
 # --- SECTION 8: Granular HIV Heatmap ---
 st.subheader("8. Granular HIV Trends by Region Over Time")
 try:
     hiv_heatmap_data = df[['date', 'region', 'hiv_incidence']]
     hiv_heatmap_data = hiv_heatmap_data.groupby(['region', 'date'])['hiv_incidence'].mean().reset_index()
     heatmap_pivot = hiv_heatmap_data.pivot(index='region', columns='date', values='hiv_incidence')
     fig_hiv = px.imshow(heatmap_pivot, labels=dict(x="Date", y="Region", color="HIV Incidence"), aspect='auto', 
                         color_continuous_scale='Viridis', title="Granular View: Monthly HIV Incidence by Region (1970–2020)")
     fig_hiv.update_layout(width=1000, height=700, xaxis=dict(tickangle=-45, nticks=25), yaxis=dict(autorange="reversed"))
     st.plotly_chart(fig_hiv, use_container_width=True)
 except Exception as e:
     st.error(f"Failed to generate HIV heatmap: {e}")
 
 # --- FOOTER ---
 st.markdown("---")
 st.markdown("🌐 Developed by **Valentine Ghanem** | MSc Public Health & Data Science")
