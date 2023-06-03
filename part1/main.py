from pyspark.sql import SparkSession
import re
import json

spark = SparkSession.builder.appName('ChiSquared').getOrCreate()
sc = spark.sparkContext

dataset_path = '../reviews_devset_full.json'
stopwords_path = '../stopwords.txt'

regex = re.compile(r'[ \t\d()[\]{}.!?,;:+=\-_"\'`~#>^@&*%€$§\\/]+')
stopWords = set(sc.textFile(stopwords_path).collect())
rdd_data = sc.textFile(dataset_path).map(lambda x: json.loads(x))

# counts tokens occurrences throughout all documents, the result is a dictionary that looks like this: "{'you': 4462, 'gift': 1642, 'cuisine': 19, 'page': 1441...}"
token_total_counts = dict(rdd_data.flatMap(
    lambda x: [(x['asin'], word.lower()) for word in set(regex.split(x['reviewText']))])
        .filter(lambda x: x[1] and x[1] not in stopWords)
        .map(lambda x: (x[1], x[0])) \
        .groupByKey() \
        .map(lambda x: (x[0], (len(set(x[1]))))).collect())

# Count the tokens for each category
category_tokens_counts = rdd_data.flatMap(
    lambda x: [(x['category'], x['asin'], word.lower()) for word in set(regex.split(x['reviewText']))])\
    .filter(lambda x: x[2] and x[2] not in stopWords)\
    .map(lambda x: ((x[0], x[2]), x[1])) \
    .groupByKey() \
    .map(lambda x: (x[0][0], (x[0][1], len(set(x[1]))))) \
    .groupByKey()
# we end up with an RDD that contains elements like this
#[
#('category_name', ('token_name', 'total_token_ocurrences_in_category'))
#('category_name', ('token_name2', 'total_token_ocurrences_in_category'))
#('category_name2', ('token_name', 'total_token_ocurrences_in_category'))
#...]

# Count the documents for each category
categories_document_counts = dict(rdd_data.map(lambda x: (x['category'], x['asin'])) \
                                  .groupByKey() \
                                  .map(lambda x: (x[0], len(x[1]))) \
                                  .collect())
# we end up with a dictionary that looks like this
#{'Patio_Lawn_and_Garde': 994,
# 'Apps_for_Android': 2638,
# 'Book': 22507,
# 'Sports_and_Outdoor': 3269
# ...}

# Total number of documents
N = rdd_data.count()


def calculate_chi_squared(category):
    # c is referred to the category, t is referred to the token(word)
    # In order to be able to calculate chi-square, we need to have the following values for each token:
    # N- total number of retrieved documents
    # A- number of documents in c which contain t
    # B- number of documents not in c which contain t
    # C- number of documents in c without t - this can be derived from getting the total number of documents for the category and subtracting A from it
    # D- number of documents not in c without t
    # the formula for calculating chi-squared is
    # N(AD - BC)^2 / (A+B)(A+C)(B+D)(C+D)
    category_name = category[0]

    token_chi = {}
    for token, token_count_in_category in category[1]:
        A = token_count_in_category
        B = token_total_counts[token] - A
        C = categories_document_counts[category_name] - A
        D = N - categories_document_counts[category_name] - B
        R: float = (N * (((A * D) - (B * C)) ** 2)) / ((A + B) * (A + C) * (B + D) * (C + D))
        token_chi[token] = R

    top_tokens = sorted(token_chi.items(), key=lambda x: x[1], reverse=True)[:75]

    top_tokens_str = f'<{category_name}> {" ".join([f"{token}:{chi_value}" for token, chi_value in top_tokens])}\n'

    return top_tokens_str


# Calculate chi-squared values for each category and word
chi_squared_strings = category_tokens_counts.map(calculate_chi_squared)
with open("output_rdd.txt", 'w') as f:
    for category_row in sorted(chi_squared_strings.collect()):
        f.write(category_row)
        print(category_row)

all_tokens = list(token_total_counts.keys())
all_tokens.sort()
with open('output_rdd.txt', 'a') as f:
    f.write(" ".join(all_tokens))
