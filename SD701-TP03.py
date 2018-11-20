# Databricks notebook source
''''
WHAT YOU NEED TO KNOW BEFORE READING THIS NOTEBOOK:

I am Alexis de Ipanema, and I have been having health issues from 18 October until now. I have indeed had two nose surgeries (18 October and 7 November) which led both time to an infection as well as breathing issues. I have done my best to do all I could given my extreme weakness state during this whole time, and it was unfortunately not so much for this Kaggle. For this Kaggle, I have just been running a crossvalidation for the logistic regression and a Regression tree.



# COMMAND ----------

from pyspark import SparkContext, SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

""""
Here we split the available data into a training set and a test set

# COMMAND ----------

train_data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferSchema='true').load('/FileStore/tables/train_set-51e11.csv')

test_data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferSchema='true').load('/FileStore/tables/test_set-b5f57.csv')

(trainingSet, testSet) = train_data.randomSplit([0.6, 0.4])

display(train_data)

# COMMAND ----------

print('Train data size: {} rows, {} columns'.format(train_data.count(), len(train_data.columns)))
print('Test data size: {} rows, {} columns'.format(test_data.count(), len(test_data.columns)))
print('Train set size: {} rows, {} columns'.format(trainingSet.count(), len(trainingSet.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC We will use the `VectorAssembler()` to merge our feature columns into a single vector column as requiered by Spark methods.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

vector_assembler = VectorAssembler(inputCols=["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points", "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9", "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14", "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29", "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34", "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40"], outputCol="features")


# COMMAND ----------

# MAGIC %md For this example, we will use `Logistic Regression`.

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# Setup classifier
classifier = LogisticRegression(labelCol="Cover_Type", featuresCol="features")
classifier.explainParams()


# COMMAND ----------

# MAGIC %md Here is the pipeline and the simple logistic regression. We train the model on trainingSet and we test it on test_set predictions. The accuracy is measured by the f1 measurement.

# COMMAND ----------

from pyspark.ml import Pipeline

# Chain vecAssembler and classificaiton model 
pipeline = Pipeline(stages=[vector_assembler, classifier])

# Run stages in pipeline with the train data
model = pipeline.fit(trainingSet)

evaluator = MulticlassClassificationEvaluator(labelCol="Cover_Type", predictionCol="prediction", metricName="f1")
accuracy = evaluator.evaluate(testSet_predictions)

# COMMAND ----------

'''

Here we do a cross validation on the logistic Regression with a grid of parameter to evaluate. I had initially set-up a lot more parameters to be explored, especially the elastic net that I wanted to change but when I was running the cross-validation, it was taking to much time, my PC could never end the computation. Hence I have been running this cross-validation  just for the exercize, with an ad-hoc list of paremeters, to check-out if I was able to run it down correctly.

# COMMAND ----------

#Make predictions on testSet

#testSet_predictions = model.transform(testSet)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

paramGrid = ParamGridBuilder().addGrid( classifier.maxIter, [100,200]).addGrid(classifier.regParam, [0.1, 0.01]).build()

mon_evaluateur = MulticlassClassificationEvaluator(labelCol="Cover_Type", predictionCol="prediction", metricName="f1")
    
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=mon_evaluateur)

cvModel = crossval.fit(trainingSet)

testSet_predictions = cvModel.transform(testSet)

evaluator = MulticlassClassificationEvaluator(labelCol="Cover_Type", predictionCol="prediction", metricName="f1")
accuracy_logistique_regression_CV = evaluator.evaluate(testSet_predictions)

# COMMAND ----------

'''

It turned out that the set of parameters used for this Cross-validation was not even as good as the basic parameters for the logic regression as we can see than we did slightly worse on "accuracy_logistique_regression_CV" than  "accuracy".

# COMMAND ----------

accuracy_logistique_regression_CV

# COMMAND ----------

accuracy

# COMMAND ----------

'''

I have then been running a decision tree. This run is just the last run that I have been performing. I have just been playing manually with two parameters. At first I have been increasing the number maxdepth of the tree, which seemed to me to be linked with the maximum complexity of the classifier model. We just saw the actual meaning of this parameter as we have just had or first course explaining what is the randome tree algorithm this morning. So I found-out that increasing the maxDepth was increasing the accuracy of the model until it was not increasing anymore, around 25. Then I did the same with maxBins which was to my mind linked with the granularity of the mdel and therefore somehow linked to the max complexity of the model that the user was ready to look for. I finally reached around 89 % accuracy. I am 100% sure that if I had been playing more with the parameters, I could have increased more the accuracy. But I must say that "pressing all the buttons combinations until I get the best accuracy" on an algorithm that at the time I did not even know how it worked, did not feel like it was making me learning anything so I stopped there. So here is the end of my work, I just give the accuracy two code blocks below.

# COMMAND ----------

#decision tree
from pyspark.ml.regression import DecisionTreeRegressor

#vector_assembler2 = VectorAssembler(inputCols=["Cover_Type"], outputCol="label")

dt = DecisionTreeClassifier(featuresCol="features",labelCol="Cover_Type",maxDepth=25,maxBins=64)

pipeline = Pipeline(stages=[vector_assembler, dt])

model = pipeline.fit(trainingSet)

testSet_predictions = model.transform(testSet)

evaluator = MulticlassClassificationEvaluator(labelCol="Cover_Type", predictionCol="prediction", metricName="f1")

accuracy_random_tree = evaluator.evaluate(testSet_predictions)


# COMMAND ----------

dt.explainParams()

# COMMAND ----------

'''

So here is the final accuracy that I got for my decisiont tree with maxDepth = 25 and maxBins = 64

# COMMAND ----------

accuracy_random_tree

# COMMAND ----------

# Make predictions on testData
predictions = model.transform(test_data)


predictions = predictions.withColumn("Cover_Type", predictions["prediction"].cast("int"))  # Cast predictions to 'int' to mach the data type expected by Kaggle
# Show the content of 'predictions'
predictions.printSchema()


# COMMAND ----------

# Display predictions and probabilities
display(predictions.select("Cover_Type", "probability"))

# COMMAND ----------

# MAGIC %md Finally, we can create a file with the predictions.

# COMMAND ----------

# Select columns Id and prediction
(predictions
 .repartition(1)
 .select('Id', 'Cover_Type')
 .write
 .format('com.databricks.spark.csv')
 .options(header='true')
 .mode('overwrite')
 .save('/FileStore/kaggle-submission'))

# COMMAND ----------

# MAGIC %md To be able to download the predictions file, we need its name (`part-*.csv`):

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/FileStore/kaggle-submission"))

# COMMAND ----------

# MAGIC %md Files stored in /FileStore are accessible in your web browser via `https://<databricks-instance-name>.cloud.databricks.com/files/`.
# MAGIC   
# MAGIC For this example:
# MAGIC 
# MAGIC https://community.cloud.databricks.com/files/kaggle-submission/part-*.csv?o=######
# MAGIC 
# MAGIC where `part-*.csv` should be replaced by the name displayed in your system  and the number after `o=` is the same as in your Community Edition URL.
# MAGIC 
# MAGIC 
# MAGIC Finally, we can upload the predictions to kaggle and check what is the perfromance.

# COMMAND ----------


