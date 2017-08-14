from pyspark import SparkContext, SparkConf
from pprint import pprint

sc = SparkContext(conf=SparkConf())


################################################################################
# aggregate
def seqOp(x, y):
    print('seqOp', x, y, sep='\t')
    return x[0] + y, x[1] + 1

def combOp(x, y):
    print('combOp', x, y, sep='\t')
    return x[0] + y[0], x[1] + y[1]

sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9]).aggregate((0, 0), seqOp, combOp)
# 因为我的电脑是四核, 所以分了四个 partition
# 输出结果如下:
# seqOp	(0, 5)	1
# seqOp	(1, 6)	2
# seqOp	(0, 5)	5
# seqOp	(5, 6)	6
# seqOp	(0, 5)	7
# seqOp	(7, 6)	8
# seqOp	(15, 7)	9
# seqOp	(0, 5)	3
# seqOp	(3, 6)	4
# combOp	(0, 5)	(3, 7)
# combOp	(3, 12)	(7, 7)
# combOp	(10, 19)	(11, 7)
# combOp	(21, 26)	(24, 8)
# (45, 34)
################################################################################
signPrefixes = loadCallSignTable()
def processSignCount(sign_count, signPrefixes):
    country = lookupCountry(sign_count[0], signPrefixes)
    count = sign_count[1]
    return (country, count)
countryContactCounts = (contactCounts
                        .map(processSignCount)
                        .reduceByKey((lambda x, y: x+ y)))

signPrefixes = sc.broadcast(loadCallSignTable())
def processSignCount(sign_count, signPrefixes):
    country = lookupCountry(sign_count[0], signPrefixes.value)
    count = sign_count[1]
    return (country, count)
countryContactCounts = (contactCounts
                        .map(processSignCount)
                        .reduceByKey((lambda x, y: x+ y)))
countryContactCounts.saveAsTextFile(outputDir + "/countries.txt")


################################################################################
# combineByKey and aggregateByKey
x = sc.parallelize([("a", 1), ("b", 4), ("c", 7), ('a', 10),
                    ("a", 2), ("b", 5), ("c", 8), ('a', 11),
                    ("a", 3), ("b", 6), ("c", 9), ('a', 12)], 2)


def seqOp(x, y):
    print('seqOp called: x={}, y={}'.format(x, y))
    return x[0] + y, x[1] + 1

def combOp(x, y):
    print('combOp called: x={}, y={}'.format(x, y))
    return x[0] + y[0], x[1] + y[1]

print(x.aggregateByKey((0, 0), seqOp, combOp).collect())


def createCombiner(x):
    print('createCombiner', x, sep='\t')
    return (x, 1)

def mergeValue(x, y):
    print('mergeValue', x, y, sep='\t')
    return x[0] + y, x[1] + 1

def mergeCombiner(x, y):
    print('mergeCombiner', x, y, sep='\t')
    return x[0] + y[0], x[1] + y[1]

print(x.combineByKey(createCombiner, mergeValue, mergeCombiner).collect())
################################################################################


################################################################################
# page rank
import numpy as np
n = 10
a = np.random.randint(0, 2, (n, n))

pageLink = sc.parallelize([(i, j) for i in range(n) for j in range(n)
                           if i != j and a[i, j]]).groupByKey().persist()
print(pageLink.mapValues(list).collect())
pageRank = pageLink.mapValues(lambda x: 1.0)


def contribute(item):
    _, (links, rank) = item
    return list(map(lambda x: (x, rank / len(links)), links))


for i in range(10):
    contributions = pageLink.join(pageRank).flatMap(contribute)
    pageRank = contributions.reduceByKey(lambda x, y: x + y).mapValues(
        lambda x: 0.15 + 0.85 * x)


# def f(splitIndex, iterator):
#     for v in iterator:
#         yield splitIndex, v
################################################################################



################################################################################
# 遗传算法, 通过变换矩阵中元素的位置, 使得第一行的元素之和最大
import numpy as np
import random
import heapq
import time

npop, nrow, ncol = 100, 10, 100
origin = np.random.randint(10, 99, (1, npop, nrow, ncol))


def crossover(x):
    """调换两列(除第一行)的位置"""
    c1, c2 = np.random.permutation(ncol)[:2]
    x[1:, [c1, c2]] = x[1:, [c2, c1]]


def mutate(x):
    """调换第一行某个元素和相应列的其他行元素"""
    r = random.randint(1, nrow - 1)
    c = random.randint(1, ncol - 1)
    x[[0, r], c] = x[[r, 0], c]


def get_score(x):
    return np.sum(x[0])


def optimize(x, p_elite=0.25, max_iter=20, mutate_rate=0.8):
    scores = np.array(list(map(get_score, x)))
    n = int(npop * p_elite)
    inew = random.randint(1, n)
    for epoch in range(max_iter):
        while inew < npop:
            iold = random.randint(0, min(n, inew) - 1)
            x[inew] = x[iold]
            if random.uniform(0, 1) < mutate_rate:
                mutate(x[inew])
            else:
                crossover(x[inew])
            scores[inew] = get_score(x[inew])
            inew += 1
        idx = np.argsort(-scores)
        scores = scores[idx]
        x[:n] = x[idx[:n]]
        #print('epoch: {:2}, best_scores: {}'.format(epoch, scores[0]))
        inew = n
    return scores[:n]


def optimize_partition(iterator):
    top100 = []
    unq = 0
    for data in iterator:
        scores = optimize(data, max_iter=20)
        for i, s in enumerate(scores):
            unq += 1
            if len(top100) < 100:
                heapq.heappush(top100, (s, unq, data[i]))
            else:
                if s > top100[0][0]:
                    heapq.heappushpop(top100, (s, unq, data[i]))
                else:
                    break
    yield from top100


tic = time.time()
nn = 16
data_rdd = sc.parallelize([d for d in origin.repeat(nn, axis=0)])
for i in range(10):
    temp = sorted(data_rdd.mapPartitions(optimize_partition).collect(),
                  key=lambda x: (-x[0], x[1]))
    best_score = temp[0][0]
    temp = np.concatenate([d.reshape(1, nrow, ncol) for _, _, d in temp[:100]])
    data_rdd = sc.parallelize([temp for _ in range(nn)])
    print('epoch: {}, cost: {}s, best_score: {}'.format(i, time.time() - tic, best_score))



data = origin[0].copy()
tic = time.time()
for i in range(40):
    scores = optimize(data)
    print('epoch: {}, cost: {}s, best_score: {}'.format(i, time.time() - tic, scores[0]))



########################################################################
# 温故
########################################################################
rdd = sc.parallelize(range(10))
rdd2 = sc.parallelize([('a', 1), ('b', 2), ('a', 3), ('b', 4), ('a', 5)])

print(rdd2.mapValues(lambda x: (1, x)).union(rdd2.mapValues(lambda x: (2, x))).groupByKey().mapValues(list).collect())
print(rdd2.cogroup(rdd2).mapValues(lambda x: [list(y) for y in x]).collect())

rdd3 = sc.parallelize([('a', 1)])
print(rdd2.subtractByKey(rdd3).collect())
def filter_func(pair):
    key, (val1, val2) = pair
    return val1 and not val2
print(rdd2.cogroup(rdd3).filter(filter_func).mapValues(lambda x: x[0]).collect())
print(rdd2.cogroup(rdd3).filter(filter_func).flatMapValues(lambda x: x[0]).collect())

def collect_partition(i):
    def func(j, iterator):
        if j < i:
            # yield from iterator
            yield list(iterator)
    return func

print(rdd.mapPartitionsWithIndex(collect_partition(2), True).collect())




def dispatch(seq):
    vbuf, wbuf = [], []
    for (n, v) in seq:
        if n == 1:
            vbuf.append(v)
        elif n == 2:
            wbuf.append(v)
    return ((v, w) for v in vbuf for w in wbuf)

rdd2.mapValues(lambda x: (1, x)).union(rdd2.mapValues(lambda x: (2, x))).groupByKey().flatMapValues(dispatch).collect()



tmp = sc.parallelize([('a', 1), ('b', 2), ('1', 3), ('d', 4), ('2', 5)])
print(tmp.sortByKey().collect())


a = sc.accumulator(1)
#rdd.foreach(lambda x: a.add(x))
#rdd.foreach(lambda x: a = a + x)
#rdd.foreach(lambda x: a += x)
print(a.value)



import os
import numpy as np
from pyspark.mllib.classification import LogisticRegressionWithSGD


spark_home = '/home/wfj/spark-2.0.2-bin-hadoop2.7/data/mllib/'

# RDD based api 要预测的值放在第一列
# dataframe based api

########################################################################
# Binary Classification
########################################################################
# Load and parse the data
data = sc.textFile(os.path.join(spark_home, 'sample_svm_data.txt'))
parsedData = data.map(lambda line: np.array([float(x) for x in line.split(' ')]))
model = LogisticRegressionWithSGD.train(parsedData)

# Build the model
labelsAndPreds = parsedData.map(lambda point: (int(point.item(0)),
        model.predict(point.take(range(1, point.size)))))

# Evaluating the model on training data
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))

########################################################################
# Clustering
########################################################################
from pyspark.mllib.clustering import KMeans
from math import sqrt

# Load and parse the data
data = sc.textFile(os.path.join(spark_home, 'kmeans_data.txt'))
parsedData = data.map(lambda line: np.array([float(x) for x in line.split(' ')]))

# Build the model (cluster the data)
clusters = KMeans.train(parsedData, 2, maxIterations=10)
        #runs=30 initialization_mode="random")

# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(np.sum((point - center) ** 2))
    #return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))


########################################################################
# SparkSession
#
# dataframe, transformer, estimator, pipeline
########################################################################
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# df = spark.read.json('/home/wfj/spark-2.0.2-bin-hadoop2.7/examples/src/main/resources/people.json')
# df.createOrReplaceTempView("people")
# sqlDF = spark.sql("SELECT * FROM people") # 直接运行 sql 语句
# sqlDF.show()

# 使用 pipeline
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer

# Prepare training documents from a list of (id, text, label) tuples.
training = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0)
], ["id", "text", "label"]) # NOTE column names, 下面会用到

# Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

# Fit the pipeline to training documents.
model = pipeline.fit(training)

# Prepare test documents, which are unlabeled (id, text) tuples.
test = spark.createDataFrame([
    (4, "spark i j k"),
    (5, "l m n"),
    (6, "spark hadoop spark"),
    (7, "apache hadoop")
], ["id", "text"])

# Make predictions on test documents and print columns of interest.
prediction = model.transform(test) # NOTE is transform, no predict
selected = prediction.select("id", "text", "probability", "prediction")
for row in selected.collect():
    rid, text, prob, prediction = row
    print("(%d, %s) --> prob=%s, prediction=%f" % (rid, text, str(prob), prediction))
