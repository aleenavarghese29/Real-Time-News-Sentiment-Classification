import streamlit as st
from newsapi import NewsApiClient
from textblob import TextBlob
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType, StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
import pandas as pd
import plotly.express as px


st.title("Real-Time News Sentiment Dashboard (PySpark)")

# -----------------------
# Initialize Spark and NewsAPI
# -----------------------
spark = SparkSession.builder.appName("NewsSentimentAPI").getOrCreate()
api = NewsApiClient(api_key='eda1460cb9cc4cbebb8dab34df78501c')



# -----------------------
# Fetch news from API
# -----------------------
def fetch_news_api():
    articles = api.get_top_headlines(language='en', page_size=50)['articles']
    news_data = [{"title": a['title'], "description": a.get('description','')} for a in articles if a['title']]
    return pd.DataFrame(news_data)

# -----------------------
# Label sentiment using TextBlob (for training only)
# -----------------------
def label_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 2  # Positive
    elif polarity < -0.1:
        return 0  # Negative
    else:
        return 1  # Neutral

# -----------------------
# Train PySpark ML pipeline
# -----------------------
def train_model(df):
    sdf = spark.createDataFrame(df)
    sdf = sdf.withColumn("label", udf(label_sentiment, IntegerType())(col("title")))

    tokenizer = Tokenizer(inputCol="title", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label")

    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])
    model = pipeline.fit(sdf)
    return model

# -----------------------
# Predict sentiment
# -----------------------
def predict_sentiment(model, df):
    sdf = spark.createDataFrame(df)
    predictions = model.transform(sdf)
    
    # Map numeric labels to string
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    predictions = predictions.withColumn(
        "Sentiment",
        udf(lambda x: label_map[x], StringType())(col("prediction"))
    )
    return predictions.select("title", "description", "Sentiment").toPandas()

# -----------------------
# Main
# -----------------------
df_news = fetch_news_api()
if not df_news.empty:
    model = train_model(df_news)
    df_predictions = predict_sentiment(model, df_news)

    # Show table
    st.subheader("Latest News Headlines with Sentiment")
    st.dataframe(df_predictions)

    # Show bar chart
    st.subheader("Sentiment Distribution")
    sentiment_counts = df_predictions['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig = px.bar(sentiment_counts, x='Sentiment', y='Count',
                 color='Sentiment', color_discrete_map={"Positive":"green","Neutral":"gray","Negative":"red"},
                 text='Count', title="Sentiment Distribution of Latest News Headlines")
    fig.update_layout(yaxis=dict(dtick=1))
    st.plotly_chart(fig)
    
import time

refresh_interval = 60  # seconds
time.sleep(refresh_interval)
st.experimental_rerun()
