from __future__ import print_function
from pyspark.ml.feature import PCA, VectorAssembler
from pyspark.mllib.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.mllib.feature import Normalizer
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde

sc = SparkContext("local", "pca-app")

sqlContext = SQLContext(sc)

normalizer = Normalizer()

df = sqlContext.read.json("/tmp/iris.json")
vecAssembler = VectorAssembler(inputCols=["sepal_width", "sepal_length", "petal_width", "petal_length"], outputCol="features")

pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")

pipeline = Pipeline(stages=[vecAssembler, pca])

model = pipeline.fit(df)
xy = model.transform(df).select("pcaFeatures").map(lambda row: [row[0][0], row[0][1]]).collect()

# convert to numpy array so we can easily transpose later
xy = np.array(xy)

x=np.array(zip(*xy)[0])
y=np.array(zip(*xy)[1])

plt.scatter(x,y)
plt.show()

plt.hexbin(x,y,  gridsize=30)
plt.show()

nbins = 55


k = kde.gaussian_kde(np.transpose(xy))
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
plt.show()
