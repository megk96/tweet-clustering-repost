# Clustering Repost Tweets
This assignment has the following two tasks.  
1. Finding the “true” number of posts, by grouping the Twitter posts together if they are retweeting or commenting on the same message. Your method should be robust to small changes, so that if someone introduces a small typo or uses a retweet prefix or postfix that’s not in the set of data provided, your technique for grouping would still work. 
2. Ranking these groups of Twitter posts by popularity.

## Tweet Clustering

### Data Cleaning
1. All URLs are removed. 
2. All letters are converted to lower-case. 
3. All punctuation is removed. 
4. Lemmatization using port stemmer by retaining the root of the word. 
5. All stop words are removed. 

### Similarity Measure
1. The TF-IDF index is built. 
2. The cosine similarity is calculated. 

### Clustering Algorithm

The Hierarchical Clustering Algorithm is used to generate the clusters using the similarity calculated from TF-IDF. 
The clusters are stored in the file output_tweets.json

## Popularity Ranking
Intuitively, the most popular tweets are the ones that are retweeted maximum times. Although, there is another way of looking at it, based on outreach. 
### Outreach
The collective sum of the follower count of each of the authors in that cluster would give the outreach. 
In the real world, there will be followers that are repeated, but in this current dataset, that information cannot be determined, so a sum total is considered. 

### Retweets
Number of retweets can be an important determinant of popularity, because it measures how popular the tweet is, rather than how popular the author of the tweet is. 


### Retweetability
Another interesting metric is how Interesting a tweet is, which can be determined by the number of retweets per outreach. If more people are likely to retweet it, it is more interesting, with a higher retweetability.  

All three of these metrics are explored and results stored in results.txt
