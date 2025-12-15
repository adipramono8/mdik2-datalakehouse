-- Analysis: Correlation between Daily Sales and Negative Tweets
-- Author: Your Name
-- Date: 2025-11-27

SELECT 
    sales.order_date, 
    sales.category, 
    SUM(sales.amount) as total_revenue,
    COUNT(tweets.tweet_id) as complaint_count
FROM read_parquet('s3://lakehouse/gold/sales_daily.parquet') as sales
JOIN read_parquet('s3://lakehouse/gold/tweets_sentiment.parquet') as tweets
    ON sales.order_date = tweets.created_date
WHERE tweets.sentiment_score < 0
GROUP BY 1, 2
ORDER BY 1 DESC;