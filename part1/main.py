from pyspark import Row
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, RegexTokenizer, CountVectorizer, StringIndexer, \
    ChiSqSelector

from pyspark.sql.functions import col, collect_list, explode, count

stopwords = []
with open('../stopwords.txt', 'r') as file:
    for word in file.read().split("\n"):
        stopwords.append(word)

spark = SparkSession.builder.appName("ReviewTokenization") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

df = spark.read.json("../reviews_devset_small.json")
pattern = r'\b\w+\b'

tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="tokens", pattern="\\W")
df_tokenized = tokenizer.transform(df)

stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=stopwords)
categories_token_counts = stopwords_remover.transform(df_tokenized)\
    .select(col("category"), col("filtered_tokens").alias("tokens"))
    # .show(n=10000)

# Convert tokens to numerical features using CountVectorizer
count_vectorizer = CountVectorizer(inputCol="tokens", outputCol="features")
count_vectorizer_model = count_vectorizer.fit(categories_token_counts)
df_features = count_vectorizer_model.transform(categories_token_counts)

# Convert category to numerical labels using StringIndexer
label_indexer = StringIndexer(inputCol="category", outputCol="label")
df_labeled = label_indexer.fit(df_features).transform(df_features)

# Convert DataFrame to RDD for MLlib's ChiSqSelector
rdd_data = df_labeled.select(col("label"), col("features")).rdd.map(lambda row: Row(label=row.label, features=row.features))

df_rdd = spark.createDataFrame(rdd_data)

# Perform chi-squared selection
selector = ChiSqSelector(numTopFeatures=10, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
selector_model = selector.fit(df_rdd)
df_selected = selector_model.transform(df_rdd)

# Convert back to DataFrame for further analysis or display
df_result = df_selected.select(col("label"), col("selectedFeatures").alias("features"))

# Show the resulting DataFrame with selected features
df_result.show(truncate=False, n=1000)