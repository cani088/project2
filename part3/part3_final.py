import json
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover, ChiSqSelector, IDF, Normalizer
from pyspark.sql import SparkSession
from pyspark.ml.classification import OneVsRest
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create a SparkSession
spark = SparkSession.builder.master("local[1]").appName("part15").getOrCreate()


# Read the JSON file
#file = open("./reviews_devset.json")
#reviews_devset = file.readlines()
#reviewsDF = pd.DataFrame.from_records(list(map(json.loads, reviews_devset)))

# Read the JSON file
reviews_devset = spark.read.json("hdfs:///user/dic23_shared/amazon-reviews/full/reviews_devset.json")
reviewsDF = reviews_devset.toPandas()

# Categorize the labels
reviewsDF["category"] = pd.Categorical(reviewsDF["category"])
reviewsDF["labels"] = reviewsDF["category"].cat.codes



# Create a Spark DataFrame
spark_df = spark.createDataFrame(reviewsDF.loc[:, ["reviewText", "category", "labels"]])

# Define the pipeline stages
tokenizer = Tokenizer(inputCol="reviewText", outputCol="rawWords")
remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="words", caseSensitive=False)
hashingTF = HashingTF(inputCol=remover.getOutputCol(), outputCol="rawFeatures", numFeatures=2000)
idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features")
selector = ChiSqSelector(numTopFeatures=2000, featuresCol=idf.getOutputCol(), outputCol="selectedFeatures", labelCol="labels")
normalizer = Normalizer(inputCol="selectedFeatures", outputCol="normalizedFeatures", p=2.0)

# Define the classifier
classifier = LinearSVC()

# Define the OneVsRest classifier
oneVsRest = OneVsRest(classifier=classifier, labelCol="labels", featuresCol="normalizedFeatures", predictionCol="prediction")

# Create the pipeline
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, selector, normalizer, oneVsRest])

# Split the data into training, validation, and test sets
(trainingData, validationData, testData) = spark_df.randomSplit([0.6, 0.2, 0.2], seed=42)

# Define the parameter grid
# paramGrid = ParamGridBuilder() \
#      .addGrid(selector.numTopFeatures, [2000, 500, 100]) \
#      .addGrid(classifier.regParam, [0.1, 0.01, 0.001]) \
#      .addGrid(classifier.standardization, [True, False]) \
#      .addGrid(classifier.maxIter, [10, 20]) \
#      .build()

# paramGrid = ParamGridBuilder() \
#      .addGrid(classifier.regParam, [0.1, 0.01, 0.001]) \
#      .addGrid(classifier.standardization, [True, False]) \
#      .addGrid(classifier.maxIter, [10, 20]) \
#      .build()

paramGrid = ParamGridBuilder() \
     .addGrid(classifier.regParam, [0.1,0.2,0.3]) \
     .addGrid(classifier.standardization, [True,False]) \
     .addGrid(classifier.maxIter, [1,2,3]) \
     .build()

# Create the evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="f1")

# Create the cross-validator
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

# Fit the cross-validator to the training data
cvModel = crossval.fit(trainingData)

# Get the best model from cross-validation
bestModel = cvModel.bestModel


# Make predictions on the validation data
predictions = cvModel.transform(validationData)

# Evaluate the model using F1 measure
f1_score = evaluator.evaluate(predictions)
print("F1 Score:", f1_score)
