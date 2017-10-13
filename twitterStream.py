from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import matplotlib.pyplot as plt
conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
sc = SparkContext(conf=conf)
ssc = StreamingContext(sc, 10)   # Create a streaming context with batch interval of 10 sec


def main():
    ssc.checkpoint("checkpoint")
    pwords = load_wordlist("positive.txt")
    nwords = load_wordlist("negative.txt")
    counts = stream(ssc, pwords, nwords, 100)
    make_plot(counts)


def make_plot(counts):
    """
    Plot the counts for the positive and negative words for each timestep.
    Use plt.show() so that the plot will popup.
    """
    ### Making separate lists for positive and negative words
    positive_list = []
    negative_list = []
    for count in counts:
        positive_list.append(count[0][1])
        negative_list.append(count[1][1])
    maximum = max(max(positive_list), max(negative_list))

    ### Plotting and saving the plot
    timeStep_list = list(range(len(positive_list)))
    plt.xlabel('Time Step')
    plt.ylabel('Word Count')
    plt.axis([0-0.25, len(positive_list), 0, maximum+25])
    plt.plot(timeStep_list,positive_list, color='blue', linestyle='solid', marker='o', markerfacecolor='blue', markersize=5, label = 'positive')
    plt.plot(timeStep_list, negative_list, color='green', linestyle='solid', marker='o', markerfacecolor='green', markersize=5, label = 'negative')
    plt.legend(loc='upper left')
    plt.savefig('plot.png')

def load_wordlist(filename):
    """ 
    This function should return a list or set of words from the given filename.
    """
    ### Reading the file and returning a list of words in the file
    words = sc.textFile(filename,)
    word_list = words.flatMap(lambda x: x.encode("utf-8").split('\n'))
    words_list = word_list.collect()
    return words_list


def stream(ssc, pwords, nwords, duration):
    kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
    tweets = kstream.map(lambda x: x[1].encode("ascii","ignore"))
    # Each element of tweets will be the text of a tweet.
    # You need to find the count of all the positive and negative words in these tweets.
    # Keep track of a running total counts and print this at every time step (use the pprint function).
    
    words = tweets.flatMap(lambda x: x.split(' '))
    ### Getting the count according to class of the words(negative or positive)
    def returnClass(word):
        if word in pwords:
            return "positive"
        elif word in nwords:
            return "negative"
    wordClassCounts = words.map(lambda word: (returnClass(word.lower()), 1)).reduceByKey(lambda x, y: x + y).filter(lambda x:(x[0] == "positive" or x[0] == "negative"))

    ### Add the new values with the previous running count to get the new count
    def updateFunction(newValues, runningCount):
        if runningCount is None:
            runningCount = 0
        return sum(newValues, runningCount)
    runningCounts = wordClassCounts.updateStateByKey(updateFunction)
    runningCounts.pprint()
    # Let the counts variable hold the word counts for all time steps
    # You will need to use the foreachRDD function.
    # For our implementation, counts looked like:
    #   [[("positive", 100), ("negative", 50)], [("positive", 80), ("negative", 60)], ...]
    counts = []
    # YOURDSTREAMOBJECT.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
    wordClassCounts.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
    ssc.start()                         # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)

    return filter(None, counts)


if __name__=="__main__":
    main()
