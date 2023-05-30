import pyspark
import json
import re
import math

from pyspark.sql import SparkSession
from operator import add

spark = SparkSession.builder.appName('ChiSquared').getOrCreate()
sc = spark.sparkContext

dataset_path = '../reviews_devset_full.json'
stopwords_path = '../stopwords.txt'

regex = re.compile(r'[ \t\d()[\]{}.!?,;:+=\-_"\'`~#@&*%€$§\\/]+')

stopWords = set(sc.textFile(stopwords_path).collect())
rdd_data = sc.textFile(dataset_path).map(lambda x: json.loads(x))

# counts tokens occurrences throughout all documents
token_counts = dict(
    rdd_data.flatMap(lambda x: [(x['asin'], word.lower()) for word in set(regex.split(x['reviewText']))]) \
        .filter(lambda x: x[1] not in stopWords) \
        .map(lambda x: (x[1], x[0])) \
        .groupByKey() \
        .map(lambda x: (x[0], (len(set(x[1]))))).collect())

token_counts_per_category = rdd_data.flatMap(
    lambda x: [(x['category'], x['asin'], word.lower()) for word in set(regex.split(x['reviewText']))]) \
    .filter(lambda x: x[2] not in stopWords and x[2] and len(x[2]) > 1) \
    .map(lambda x: ((x[0], x[2]), x[1]))

category_tokens = token_counts_per_category.groupByKey().map(
    lambda x: (x[0][0], (x[0][1], len(set(x[1]))))) \
    .groupByKey()

categories_counts = dict(rdd_data.map(lambda x: (x['category'], x['asin'])) \
                         .groupByKey() \
                         .map(lambda x: (x[0], len(x[1]))) \
                         .collect())

N = rdd_data.count()

def calculate_chi_square(category_data):
    category = category_data[0]

    token_chi = {}
    for token, count_in_category in category_data[1]:
        A = count_in_category
        B = token_counts[token] - A
        C = categories_counts[category] - A
        D = N - categories_counts[category] - B
        try:
            R: float = (N * (((A * D) - (B * C)) ** 2)) / ((A + B) * (A + C) * (B + D) * (C + D))
        except ZeroDivisionError:
            R = 0
        token_chi[token] = R

    top_terms = sorted(token_chi.items(), key=lambda x: x[1], reverse=True)[:75]
    top_terms_str = f"<{category}>\t{' '.join([f'{term}:{score:.4f}' for term, score in top_terms])}\n"
    with open("output.txt", 'a') as f:
        f.write(top_terms_str)
    return top_terms_str


# Calculate chi-squared values for each category and word
chi_squared_values = category_tokens.map(calculate_chi_square)
result_string = "\n".join(sorted(chi_squared_values.collect()))
print(result_string)
