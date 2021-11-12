from pyspark.sql import functions, SparkSession, types
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
import os


def read_data():
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    data = spark.read.csv(__location__ + os.sep + "iris.csv", inferSchema=True, header=True)
    training, validation = data.randomSplit([0.75, 0.25])
    return training, validation


def classification():
    training, validation = read_data()
    to_categorical = StringIndexer(inputCol="variety", outputCol="var").fit(training)
    training = to_categorical.transform(training)
    validation = to_categorical.transform(validation)
    assemble_features = VectorAssembler(inputCols=["slength", 'swidth', 'plength', 'pwidth'], outputCol="features")
    classifier = RandomForestClassifier(featuresCol='features', labelCol='var')
    pipeline = Pipeline(stages=[assemble_features, classifier])

    model = pipeline.fit(training)
    predictions = model.transform(validation)
    predictions.show()

    evaluation = MulticlassClassificationEvaluator(
        predictionCol='prediction', labelCol='var', metricName='accuracy'
    )
    acc = evaluation.evaluate(predictions)
    print(acc)


if __name__ == '__main__':
    spark = SparkSession.builder.appName("Iris Data Classification").getOrCreate()
    classification()
