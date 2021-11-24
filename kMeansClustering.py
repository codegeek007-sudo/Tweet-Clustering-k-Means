import pandas as pd
import random
import re
import csv
import matplotlib.pyplot as plt

##DEBUGGING FLAG
debug = False
empStr = " "


# K-Means algorithm
def k_means(tweets, k, iterations):
    # Declarations
    centroids = []  # list of current controids
    kCount = 0
    iterationsCount = 0
    old_Centroiod = []  # list of all old centroids computed before
    initKVals = dict()

    # Assign random values to k to init
    while kCount < k:
        size = len(tweets) - 1
        randomIndex = random.randint(0, size)
        if randomIndex not in initKVals:
            kCount = kCount + 1
            initKVals[randomIndex] = True
            centroids.append(tweets[randomIndex])

    while (convergenceCheck(old_Centroiod, centroids) == False & (iterationsCount < iterations)):
        clusters = clusterize(tweets, centroids)
        old_Centroiod = centroids
        centroids = calcCentroid(clusters)
        iterationsCount += 1
    sse = calc_SSE(clusters)

    return clusters, sse


# SSE Error function
def calc_SSE(clusterSet):
    sse = 0
    for i in range(len(clusterSet)):
        for j in range(len(clusterSet[i])):
            sse = sse + (clusterSet[i][j][1] * clusterSet[i][j][1])
    return sse


# Calcuting the distance between 2 tweets using Jaccard Distance, given in the assignment file
def jDistance(firstTweet, secondTweet):
    # get the intersection
    intersection = set(firstTweet).intersection(secondTweet)
    union = set().union(firstTweet, secondTweet)
    distance = 1 - (len(intersection) / len(union))
    return distance


# Check convergence
def convergenceCheck(oldCentroids, newCentroids):
    flag = False
    if len(oldCentroids) < len(newCentroids) or len(oldCentroids) > len(newCentroids):
        return flag

    for i in range(len(newCentroids)):
        if empStr.join(newCentroids[i]) != empStr.join(oldCentroids[i]):
            flag = False
    else:
        flag = True
    return flag


def clusterize(tweets, centroids):
    clusters = dict()
    for i in range(len(tweets)):
        minDist = float('inf')
        clusterPoint = -1
        for j in range(len(centroids)):
            distance = jDistance(centroids[j], tweets[i])
            # assign new, closer, distance
            if distance < minDist:
                clusterPoint = j
                minDist = distance
            # If the centroid is equal to the input
            if centroids[j] == tweets[i]:
                clusterPoint = j
                minDist = 0
                break
        if minDist == 1:
            clusterPoint = random.randint(0, len(centroids) - 1)
        clusters.setdefault(clusterPoint, []).append([tweets[i]])
        last_tweet = len(clusters.setdefault(clusterPoint, [])) - 1
        clusters.setdefault(clusterPoint, [])[last_tweet].append(minDist)

    return clusters


def calcCentroid(clusters):
    centroids = []

    # iterate each cluster and check for a tweet with closest distance sum with all other tweets in the same cluster
    # select that tweet as the centroid for the cluster
    for i in range(len(clusters)):
        min_dis_sum = float('inf')
        centroidPoint = -1  # Initialization value
        minDistList = []

        for t1 in range(len(clusters[i])):
            minDistList.append([])
            totalDist = 0
            # get distances sum for every of tweet t1 with every tweet t2 in a same cluster
            for t2 in range(len(clusters[i])):
                if t1 != t2:
                    if t2 < t1:
                        dis = minDistList[t2][t1]
                    else:
                        dis = jDistance(clusters[i][t1][0], clusters[i][t2][0])

                    minDistList[t1].append(dis)
                    totalDist += dis
                else:
                    minDistList[t1].append(0)

            # select the tweet with the minimum distances sum as the centroid for the cluster
            if totalDist < min_dis_sum:
                min_dis_sum = totalDist
                centroidPoint = t1

        # append the selected tweet to the centroid list
        centroids.append(clusters[i][centroidPoint][0])

    return centroids


def pre_process():
    '''
    THIS IS THE ORIGINAL URL
    '''
    url = "https://raw.githubusercontent.com/codegeek007-sudo/ML/main/nytimeshealth.txt"

    '''
    THIS IS THE MODIFIED URL
    '''
    # url = "https://raw.githubusercontent.com/codegeek007-sudo/ML/main/nytimeMOD.txt"

    df = pd.read_csv(url, quoting=csv.QUOTE_NONE, error_bad_lines=False, warn_bad_lines=False)
    if (debug):
        print(df)
    tweets = df.values.tolist()
    list_tweet = []

    '''
    PRE-PROCESSING
    '''
    for i in range(len(tweets)):
        tweet = ' '.join(str(j) for j in tweets[i])
        tweet = tweet.strip('\n')
        tweet = tweet[50:]  # First 50 characters can be ignored
        # Removing word(s) after the @ symbol
        tweet = re.sub(r"@\S+", "", tweet)
        # Removing # signs
        tweet = tweet.replace('#', '')
        # Remove  URLs
        tweet = re.sub(r"http\S+", "", tweet)
        tweet = re.sub(r"www\S+", "", tweet)
        tweet = tweet.strip()
        tweet_len = len(tweet)
        # Make text lower case
        tweet = tweet.lower()
        # Remove rt
        tweet = tweet.replace('rt', '')
        # Remove :
        tweet = tweet.replace(':', '')
        # Removing whitespace
        tweet = " ".join(tweet.split())
        list_tweet.append(tweet.split(' '))
        if (debug):
            print(tweet)

    return list_tweet


if __name__ == "__main__":

    tweets = pre_process()
    '''
    Once the data is pre-processed, the k-means algo will need to execute a n times, where n is the number of trials
    '''
    # Num of trials a user wishes to have. This val is hard-coded
    trials = 5
    trialCount = 0

    # Number of clusters a user wishes to have. This val is hard-coded. This is the starting value
    value_of_K = 3

    # Number of max iterations a user wishes to have. This val is hard-coded.
    maxIterations = 200

    for n in range(trials):
        trialCount = n + 1
        print('Trial # ' + str(trialCount) + ':')
        clusters, sse = k_means(tweets, value_of_K, maxIterations)

        for i in range(len(clusters)):
            # print(clusters[i])
            # print(type(clusters[i]))
            print('Size of Cluster ' + str(i + 1) + ': ', str(len(clusters[i])) + (' tweets'))
        print("Calculated SSE: " + str(sse) + "\n")
        value_of_K = value_of_K + 3
        plt.scatter(float(value_of_K), sse)
        plt.plot(float(value_of_K), sse, linestyle='-', marker='o')
        plt.title("SSE vs. Number of Clusters")
        plt.xlabel('Number of Clusters')
        plt.ylabel('SSE')
    plt.show()


