# Streamlit imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Import additional ML libraries if needed
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor

# Set Streamlit page configuration
st.set_page_config(page_title="Anime Data Analysis", layout="wide")

# Title and description
st.title("Anime Data Analysis and Recommendation System")
st.write("""
This application allows you to explore anime data, visualize trends, and use machine learning to recommend similar anime titles.
""")

# File upload
anime_file = st.file_uploader("Upload anime dataset (anime_cleaned.csv)", type="csv")
user_file = st.file_uploader("Upload user dataset (users_cleaned.csv)", type="csv")

# Load datasets if uploaded
if anime_file and user_file:
    anime_df = pd.read_csv(anime_file)
    user_df = pd.read_csv(user_file)

    # Data overview
    st.subheader("Anime Dataset Overview")
    st.write(anime_df.head())
    st.subheader("User Dataset Overview")
    st.write(user_df.head())

    # Data cleaning and type conversion
    anime_df = anime_df.dropna()
    anime_df['episodes'] = pd.to_numeric(anime_df['episodes'], errors='coerce')
    anime_df['score'] = pd.to_numeric(anime_df['score'], errors='coerce')
    user_df['user_days_spent_watching'] = pd.to_numeric(user_df['user_days_spent_watching'], errors='coerce')

    # Scatter Plot: Score vs Episodes
    st.subheader("Scatter Plot: Anime Score vs Episodes")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(anime_df['episodes'], anime_df['score'], color='green', alpha=0.6)
    ax.set_title('Scatter Plot of Anime Score vs Episodes')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Score')
    ax.grid(True)
    st.pyplot(fig)

    # Bar Plot: Genre Distribution
    st.subheader("Genre Frequency Distribution")
    anime_genres = anime_df['genre'].str.split(',').explode().str.strip()
    genre_counts = anime_genres.value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(genre_counts.index, genre_counts.values, color='purple', alpha=0.7, edgecolor='darkviolet')
    ax.set_title('Frequency Distribution of Anime Genres')
    ax.set_xlabel('Genres')
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # Line Plot: User Days Spent Watching
    st.subheader("User Days Spent Watching Over Time")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(user_df.index, user_df['user_days_spent_watching'], color='purple', marker='o', linestyle='-', linewidth=2)
    ax.set_title('User Days Spent Watching Over Time')
    ax.set_xlabel('User Index')
    ax.set_ylabel('Days Spent Watching')
    ax.grid(True)
    st.pyplot(fig)

    # Pie Chart: Anime Watching Status Distribution
    st.subheader("Proportion of Users in Each Anime Status")
    status_counts = {
        'Watching': user_df['user_watching'].sum(),
        'Completed': user_df['user_completed'].sum(),
        'On Hold': user_df['user_onhold'].sum(),
        'Dropped': user_df['user_dropped'].sum(),
        'Plan to Watch': user_df['user_plantowatch'].sum()
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%', startangle=90,
           colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
    ax.set_title('Proportion of Users in Each Anime Status')
    st.pyplot(fig)

    # Pair Plot
    st.subheader("Pair Plot of Anime Data")
    fig = sns.pairplot(anime_df[['episodes', 'score', 'members']], diag_kind='kde', plot_kws={'alpha':0.6})
    st.pyplot(fig)

    # Cosine Similarity for Anime Recommendations
    st.subheader("Anime Recommendation Based on Similarity")
    input_anime = st.text_input("Enter an anime title for recommendations (e.g., 'One Piece')")
    if input_anime in anime_df['title'].values:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_genres = encoder.fit_transform(anime_df[['genre']])
        label_encoder = LabelEncoder()
        anime_df['rating_encoded'] = label_encoder.fit_transform(anime_df['rating'])
        features_matrix = pd.concat(
            [anime_df[['rating_encoded', 'members']], pd.DataFrame(encoded_genres)], axis=1).fillna(0)
        similarity_matrix = cosine_similarity(features_matrix)
        
        input_index = anime_df[anime_df['title'] == input_anime].index[0]
        similarity_scores = list(enumerate(similarity_matrix[input_index]))
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        top_n_anime_indices = [i[0] for i in sorted_scores[1:11]]
        top_n_anime_titles = anime_df.iloc[top_n_anime_indices]['title'].values
        st.write("Top similar anime to", input_anime, "are:")
        for title in top_n_anime_titles:
            st.write("-", title)
    else:
        st.write(f"{input_anime} not found in the dataset.")

    # Forecasting Average Anime Score with ARIMA
    st.subheader("Average Anime Score Forecast with ARIMA")
    anime_df['aired_from_year'] = pd.to_datetime(anime_df['aired_from_year'], format='%Y')
    data = anime_df.groupby('aired_from_year')['score'].mean().reset_index()
    data.set_index('aired_from_year', inplace=True)
    
    model = ARIMA(data['score'], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=5)
    forecast_index = pd.date_range(start=data.index[-1], periods=6, freq='Y')[1:]
    forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['score'], label='Historical Average Score')
    ax.plot(forecast_series.index, forecast_series, label='ARIMA Forecast', color='red')
    ax.set_title("Average Anime Score Forecast with ARIMA")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Score")
    ax.legend()
    st.pyplot(fig)

    # Prophet Forecast for Comparison
    st.subheader("Average Anime Score Forecast with Prophet")
    data_prophet = data.reset_index().rename(columns={'aired_from_year': 'ds', 'score': 'y'})
    prophet_model = Prophet(yearly_seasonality=True)
    prophet_model.fit(data_prophet)
    future = prophet_model.make_future_dataframe(periods=5, freq='Y')
    forecast = prophet_model.predict(future)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['score'], label='Historical Average Score', marker='o')
    ax.plot(forecast['ds'], forecast['yhat'], label='Prophet Forecast', color='red', marker='o')
    ax.set_title("Average Anime Score Forecast with Prophet")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Score")
    ax.legend()
    st.pyplot(fig)
else:
    st.write("Please upload both the anime and user datasets to continue.")