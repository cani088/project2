from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, StringIndexer
from pyspark.ml.linalg import Vector
from pyspark.mllib.feature import ChiSqSelector
from pyspark.sql.functions import col, explode, count
from pyspark.sql import Row

# Assuming you have a DataFrame named `df` with columns: category and reviewText

# Tokenize the reviewText column
tokenizer = Tokenizer(inputCol="reviewText", outputCol="tokens")
df_tokenized = tokenizer.transform(df)

# Define a list of stopwords
stopwords = ["the", "and", "in", "to", "of", "is", "it", "that"]  # Add or modify stopwords as needed

# Remove stopwords from the tokens
stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=stopwords)
df_filtered = stopwords_remover.transform(df_tokenized).select(col("category"), col("filtered_tokens").alias("tokens"))

# Convert tokens to numerical features using CountVectorizer
count_vectorizer = CountVectorizer(inputCol="tokens", outputCol="features")
count_vectorizer_model = count_vectorizer.fit(df_filtered)
df_features = count_vectorizer_model.transform(df_filtered)

# Convert category to numerical labels using StringIndexer
label_indexer = StringIndexer(inputCol="category", outputCol="label")
df_labeled = label_indexer.fit(df_features).transform(df_features)

# Convert DataFrame to RDD for MLlib's ChiSqSelector
rdd_data = df_labeled.select(col("label"), col("features")).rdd.map(lambda row: Row(label=row.label, features=row.features.toArray()))
df_rdd = spark.createDataFrame(rdd_data)

# Perform chi-squared selection
selector = ChiSqSelector(numTopFeatures=10, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
selector_model = selector.fit(df_rdd)
df_selected = selector_model.transform(df_rdd)

# Convert back to DataFrame for further analysis or display
df_result = df_selected.select(col("label"), col("selectedFeatures").alias("features"))

# Show the resulting DataFrame with selected features
df_result.show(truncate=False)