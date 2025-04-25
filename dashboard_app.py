
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("ghana_infectious_disease_model_dataset_cleaned.csv")

df = load_data()
df['date'] = pd.to_datetime(df['date'])

# Sidebar - Filter controls
st.sidebar.header("Filter Panel")
selected_region = st.sidebar.multiselect("Select Region(s):", df['region'].unique(), default=df['region'].unique())
selected_disease = st.sidebar.selectbox("Disease", ['hiv_incidence', 'malaria_incidence', 'tb_incidence'])
date_range = st.sidebar.date_input("Date Range", [df['date'].min(), df['date'].max()])

# Filter data
filtered_df = df[(df['region'].isin(selected_region)) & 
                 (df['date'] >= pd.to_datetime(date_range[0])) & 
                 (df['date'] <= pd.to_datetime(date_range[1]))]

# Main page
st.title("📈 Ghana Infectious Disease Trends Dashboard")
st.markdown("#### Machine Learning-Powered Epidemiology | HIV/AIDS Focus")

st.subheader("1. National Disease Trends Over Time")
fig = px.line(filtered_df, x='date', y=selected_disease, color='region', title=f"{selected_disease.replace('_', ' ').title()} Over Time")
st.plotly_chart(fig, use_container_width=True)

st.subheader("2. Regional Distribution Map")
m = folium.Map(location=[7.9, -1.0], zoom_start=6)
for _, row in filtered_df.groupby('region').tail(1).iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=10,
        color='red',
        fill=True,
        fill_opacity=0.6,
        popup=f"{row['region']}<br>{selected_disease}: {row[selected_disease]:.1f}"
    ).add_to(m)

st_data = st_folium(m, width=700)

st.subheader("3. Behavioral & Demographic Correlation")
selected_var = st.selectbox("Choose variable to compare with incidence:", 
                            ['education_access_index', 'condom_use_rate', 'urbanization_level', 
                             'hiv_awareness_index', 'youth_unemployment_rate'])

fig2 = px.scatter(filtered_df, x=selected_var, y=selected_disease, color='region',
                  title=f"{selected_var.replace('_', ' ').title()} vs. {selected_disease.replace('_', ' ').title()}")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("4. ML Forecasting Results")
st.markdown("🧠 Coming Soon: Machine learning forecasts of HIV/AIDS incidence to 2030.")

# Optional: If CSV of forecasts available
# forecast_df = pd.read_csv("hiv_forecast.csv")
# fig3 = px.line(forecast_df, x="date", y="predicted", title="Forecasted HIV Incidence")
# st.plotly_chart(fig3)
