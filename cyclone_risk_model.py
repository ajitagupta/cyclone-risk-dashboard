import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
from statsmodels.tsa.arima.model import ARIMA
import geopandas as gpd
from shapely.geometry import Point

# Generate Dummy Data
def generate_dummy_data():
    np.random.seed(42)
    dates = pd.date_range(start='1980-01-01', periods=500, freq='M')
    regions = ['Atlantic', 'Pacific', 'Indian Ocean', 'Mediterranean']
    data = {
        'Date': np.random.choice(dates, size=500),
        'Cyclone_ID': np.arange(1, 501),
        'Max_Wind_Speed': np.random.randint(100, 300, size=500),
        'Estimated_Loss': np.random.uniform(10, 500, size=500),
        'Longitude': np.random.uniform(-180, 180, size=500),
        'Latitude': np.random.uniform(-90, 90, size=500),
        'Region': np.random.choice(regions, size=500)
    }
    return pd.DataFrame(data)

# Load data
@st.cache_data
def load_data():
    return generate_dummy_data()

data = load_data()

# Sidebar Filters
st.sidebar.header("Filters")
year_range = st.sidebar.slider("Select Year Range", min_value=int(data.Date.dt.year.min()),
                               max_value=int(data.Date.dt.year.max()), value=(2000, 2023))
region = st.sidebar.selectbox("Select Region", options=["All"] + list(data['Region'].unique()))

# Filter data
filtered_data = data[(data.Date.dt.year >= year_range[0]) & (data.Date.dt.year <= year_range[1])]
if region != "All":
    filtered_data = filtered_data[filtered_data['Region'] == region]

# Title
st.title("Cyclone Risk Analysis Dashboard")
st.write("This dashboard provides statistical insights into cyclone occurrences and estimated losses.")

# Plot 1: Cyclone Frequency Over Time
st.subheader("Cyclone Frequency Over Time")
time_series = filtered_data.groupby(filtered_data.Date.dt.year)['Cyclone_ID'].count()
fig, ax = plt.subplots()
ax.plot(time_series.index, time_series.values, marker='o', linestyle='-')
ax.set_xlabel("Year")
ax.set_ylabel("Number of Cyclones")
ax.set_title("Yearly Cyclone Occurrence")
st.pyplot(fig)
st.write("This plot shows the trend of cyclone occurrences over the years. A rising trend may indicate increasing cyclone activity due to climate change, while a declining trend could suggest improved mitigation efforts or natural climate variability.")

# Plot 2: Poisson Distribution for Cyclone Frequency
st.subheader("Poisson Model: Cyclone Probability Distribution")
lambda_ = time_series.mean()
x = np.arange(0, max(time_series) + 5)
y = poisson.pmf(x, lambda_)
fig, ax = plt.subplots()
ax.bar(x, y, alpha=0.6, color='b')
ax.set_xlabel("Number of Cyclones")
ax.set_ylabel("Probability")
ax.set_title(f"Poisson Distribution (λ = {lambda_:.2f})")
st.pyplot(fig)
st.write("This Poisson distribution represents the probability of different numbers of cyclone occurrences per year. It helps in understanding the likelihood of extreme cyclone events and the expected frequency under normal conditions.")

# Plot 3: Wind Speed vs. Damage Scatter Plot
st.subheader("Wind Speed vs. Estimated Damage")
fig, ax = plt.subplots()
sns.scatterplot(data=filtered_data, x='Max_Wind_Speed', y='Estimated_Loss', alpha=0.6, ax=ax)
ax.set_xlabel("Max Wind Speed (km/h)")
ax.set_ylabel("Estimated Loss ($M)")
ax.set_title("Cyclone Wind Speed vs. Economic Loss")
st.pyplot(fig)
st.write("This scatter plot shows the relationship between cyclone wind speed and economic loss. Generally, higher wind speeds correlate with greater economic damage, highlighting the importance of infrastructure resilience in cyclone-prone areas.")

# Plot 4: Cyclone Paths on Map (Using GeoPandas)
st.subheader("Cyclone Tracks")

# Load world map from local file (Downloaded from Natural Earth Data)
world = gpd.read_file("data/ne_110m_admin_0_countries.shp")

# Convert filtered cyclone data into GeoDataFrame
geometry = [Point(xy) for xy in zip(filtered_data['Longitude'], filtered_data['Latitude'])]
geo_df = gpd.GeoDataFrame(filtered_data, geometry=geometry)

fig, ax = plt.subplots(figsize=(10, 5))
world.plot(ax=ax, color='lightgrey')
geo_df.plot(ax=ax, markersize=10, color='red', alpha=0.6, label='Cyclone Path')
ax.set_title("Global Cyclone Tracks")
ax.legend()
st.pyplot(fig)
st.write("This map displays cyclone tracks across different regions. The density of cyclone paths in specific areas helps identify high-risk zones where mitigation strategies should be prioritized.")

# Forecasting Cyclone Occurrence (ARIMA)
st.subheader("Cyclone Frequency Forecast")
model = ARIMA(time_series, order=(2,1,0))
model_fit = model.fit()
predictions = model_fit.forecast(steps=5)

fig, ax = plt.subplots()
ax.plot(time_series.index, time_series.values, marker='o', linestyle='-', label="Historical")
ax.plot(range(time_series.index[-1] + 1, time_series.index[-1] + 6), predictions, marker='o', linestyle='--', label="Forecast")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Cyclones")
ax.set_title("Cyclone Frequency Forecast (Next 5 Years)")
ax.legend()
st.pyplot(fig)
st.write("This forecast predicts cyclone occurrences over the next five years using an ARIMA model. It provides insights into future cyclone risks, helping stakeholders prepare for potential extreme weather events.")

st.write("*Disclaimer: Forecast is based on past trends and may not predict extreme climate events.*")

st.success("Dashboard ready! Select filters to explore cyclone trends and risks.")
