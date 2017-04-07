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
n = 10
a = np.random.randint(0, 2, (n, n))

pageLink = sc.parallelize([(i, j) for i in range(n) for j in range(n)
                           if i != j and a[i, j]]).groupByKey().persist()
pageRank = pageLink.mapValues(lambda x: 1.0)


def contribute(item):
    _, (links, rank) = item
    return list(map(lambda x: (x, rank / len(links)), links))


for i in range(10):
    contributions = pageLink.join(pageRank).flatMap(contribute)
    pageRank = contributions.reduceByKey(lambda x, y: x + y).mapValues(
        lambda x: 0.15 + 0.85 * x)


def f(splitIndex, iterator):
    for v in iterator:
        yield splitIndex, v
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
