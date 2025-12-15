from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, lower, when, count, sum, avg, lit, udf
from pyspark.sql.types import FloatType

# --- 1. SETUP SPARK SESSION ---
spark = SparkSession.builder \
    .appName("TGP2_Processing") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
    .config("spark.hadoop.fs.s3a.access.key", "admin") \
    .config("spark.hadoop.fs.s3a.secret.key", "password123") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

print(">>> Spark Session Created Successfully!")

# --- 2. READ RAW DATA FROM MINIO ---
print(">>> Reading Data...")
df_sales = spark.read.option("header", "true").option("delimiter", ";").csv("s3a://raw-sales-december/superstore.csv")
df_tweets = spark.read.option("header", "true").csv("s3a://raw-sales-december/twcs.csv")

# --- 3. CLEAN & PROCESS SALES DATA (Structured) ---
print(">>> Processing Sales...")
df_sales_clean = df_sales \
    .withColumn("Order Date", to_date(col("Order Date"), "MM/dd/yyyy")) \
    .groupBy("Order Date", "Category") \
    .agg(sum("Sales").alias("Total_Sales"))

# --- 4. CLEAN & PROCESS TWEETS (Unstructured) ---
# Simple Sentiment UDF
def simple_sentiment(text):
    if not text: return 0.0
    text = text.lower()
    positive_words = ["great", "good", "love", "happy", "thanks", "best", "fast"]
    negative_words = ["bad", "slow", "broken", "hate", "worst", "sad", "angry", "late"]
    score = 0
    for word in text.split():
        if word in positive_words: score += 1
        if word in negative_words: score -= 1
    return float(score)

sentiment_udf = udf(simple_sentiment, FloatType())

print(">>> Processing Tweets...")
df_tweets_clean = df_tweets \
    .withColumn("tweet_date", to_date(col("created_at"), "EEE MMM dd HH:mm:ss Z yyyy")) \
    .withColumn("text_lower", lower(col("text"))) \
    .withColumn("Derived_Category", 
        when(col("text_lower").contains("phone") | col("text_lower").contains("laptop") | col("text_lower").contains("screen"), "Technology")
        .when(col("text_lower").contains("chair") | col("text_lower").contains("table") | col("text_lower").contains("desk"), "Furniture")
        .when(col("text_lower").contains("paper") | col("text_lower").contains("binder") | col("text_lower").contains("pen"), "Office Supplies")
        .otherwise("Other")
    ) \
    .withColumn("Sentiment_Score", sentiment_udf(col("text"))) \
    .groupBy("tweet_date", "Derived_Category") \
    .agg(
        count("tweet_id").alias("Tweet_Count"),
        avg("Sentiment_Score").alias("Avg_Sentiment")
    )

# --- 5. JOIN DATASETS (Integration) ---
print(">>> Joining Datasets...")
df_final = df_sales_clean.join(
    df_tweets_clean, 
    (df_sales_clean["Order Date"] == df_tweets_clean["tweet_date"]) & 
    (df_sales_clean["Category"] == df_tweets_clean["Derived_Category"]),
    "inner"
).drop("tweet_date", "Derived_Category") 

# --- 6. WRITE TO LAKEHOUSE (Gold Layer) ---
print(">>> Writing to Data Warehouse (MinIO)...")
df_final.write.mode("overwrite").parquet("s3a://lakehouse/gold_sales_sentiment")

print(">>> SUCCESS! Pipeline Finished.")
spark.stop()