'''
Created on Dec 5, 2016

@author: weyu
'''

from __builtin__ import int
import json
import os
import sys
import time
from pyspark.context import SparkContext
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql.context import SQLContext
from pyspark.sql.types import Row, StructField, StructType, DoubleType, StringType
ascontext=None
if len(sys.argv) > 1 and sys.argv[1] == "-test":
    import os
    sc = SparkContext('local')
    sqlContext = SQLContext(sc)
    wd = os.getcwd()
    sentenceData = sqlContext.createDataFrame([
      (0, "Hi I heard about Spark"),
      (0, "I wish Java could use case classes"),
      (1, "Logistic regression models are neat")
    ], ["label", "sentence"])
    sentence = "sentence"
    numFeatures = 20
    modelpath = "/tmp/svm.model"
    modelmetadata_path = "/tmp/svm.metadata"
else:
    import spss.pyspark.runtime
    ascontext = spss.pyspark.runtime.getContext()
    sc = ascontext.getSparkContext()
    sqlContext = ascontext.getSparkSQLContext()
    sentenceData = ascontext.getSparkInputData()
    sentence = '%%input_sentence%%'
    numFeatures=int('%%input_numberFeatures%%')

tokenizer = Tokenizer(inputCol=sentence, outputCol="words")
wordsData = tokenizer.transform(sentenceData)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures = numFeatures)
featurizedData = hashingTF.transform(wordsData)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
#rescaledData.show()

schema = StructType(sentenceData.schema.fields +[
                         StructField("rawFeatures", StringType(),True),
                         StructField("features", StringType(),True)
                         ])
#rows = rescaledData.drop("words").rdd.map(lambda x:tuple([str(item) for item in x]))
#rows = rescaledData.drop("words").rdd.map(lambda x:x)
from pyspark.sql.functions import udf
from pyspark.mllib.linalg import Vectors, VectorUDT
vectorToString_udf = udf(lambda x: str(x), StringType())
stringToVector_udf = udf(lambda x: Vectors.parse(x), VectorUDT())
tmpdata = rescaledData.withColumn("rawFeatures",vectorToString_udf(rescaledData.rawFeatures)).withColumn("features",vectorToString_udf(rescaledData.features))

dfResult = sqlContext.createDataFrame(tmpdata.drop("words").rdd, schema)

if len(sys.argv) > 1 and sys.argv[1] == "-test":
    dfResult.show()
else:
    if ascontext.isComputeDataModelOnly():
        ascontext.setSparkOutputSchema(schema)
    else:
        ascontext.setSparkOutputData(dfResult)
        
# Transfer vector to common data for modeler
#schema = StructType(rescaledData.schema.fields + 
#                    [StructField("features_" + str(i),
#                                  DoubleType(), False) for i in range(1,numFeatures+1)])

#rows = rescaledData.rdd.map(lambda x: x + 
#                    tuple(x["features"].toArray().tolist()))

#dfResult = sqlContext.createDataFrame(rows, schema).drop("features").drop("words").drop("rawFeatures")

#dfResult.show()
#dfResult.withColumn("rawFeatures",stringToVector_udf(dfResult.rawFeatures)).withColumn("features",stringToVector_udf(dfResult.features)).show()
