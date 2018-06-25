# Demo of constructing recommendation system

This brief demo will walk through all the necessary steps in building up a simple recommendation system. 

This demo constructs recommendation system based upon [this popular algorithm](http://yifanhu.net/PUB/cf.pdf) with 1000+ citations, which has a Python implementation [implicit](http://implicit.readthedocs.io/en/latest/) by Ben Frederickson(@benfred).

# Getting the data

Recommentation system plays an important role in retail business and the most famous (based on Google search results) retail data I can find is an online retail data from [UCI respository](http://archive.ics.uci.edu/ml/datasets/Online+Retail). This data contains all the transactions in 01/12/2010 - 09/12/2011 for a UK-based online retailer. The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers (the data quality will be very high since the wholesaler will buy many correlated items).

# Data cleansing

First we read in the whole dataset and browse first few lines to understand the data structure:

![browsing data](/pic/data.png)

There are 4,223 unique items and 4372 unique users with half million transactions. In this demo we will just set a simply goal: try to recommend user the new item that they never bought. In doing so we can ignore the variable price and treat the quantity variable as the implicit feedback. We will also aggregate all the purchase quantity of the same item for each user, to remove the temporal information.

The formal data cleansing contains the following steps:
* Remove transactions with missing user or item (about 25% of the whole data) and negative values in quantity or price variable
* Aggregate the quantity of the same purchased items in all time
* Remove users who made transactions upon less than 10 different items. (Collaborative filtering suffers from cold-start problem, here only recommend to users we are familiar with)
* Transform user id and item description to numeric row/column index of the user-item matrix, and save it in a sparse matrix format .csr.

<img src="/pic/triplets.png" width="20%">

Above is the brief view of triplets of user-item matrix after data cleansing. The final user-item matrix has 265k user-item paris, with 3704 users and 3866 items, a sparsity of 1.85%

# Data preprocessing

To prevent overfitting and evaluate model performance, the whole dataset needs to be splitted into training and testing datasets. Since the time information is removed, we have to mask some user-item pairs in user-item matrix and move them into testing dataset. Following steps describe the details of splitting training/testing datasets:
* Randomly select 20% of users with purchases of more than 20 items
* For each selected user, randomly mask 10 items as the test data (in this way, each user in training data still have purchased at least 10 different items)

<img src="/pic/split.PNG" width="60%">

# Model fitting

The model is fitted on training dataset with the recommendation algorithm implemented in Python implicit package. To evaluate model performance, we use the metric precision at 10. For each user, this metric calculates the overlap number of between top 10 recommended items and the items actually bought in testing dataset divided by 10. This metric is self interpretable and is closely related to business application(in business the user will only pay attention to top recommendations).

Cross validation is also used to help find the optimal hyper parameters. Here we create grids of the following hyper parameters:
* scaling factor of implicit feedback (alpha)
* Number of latent factors
* regularization coefficient (lambda)
* Number of iterations

Notice that the grid search is very slow and the purpose of the demo is just a walk through instead of optimizing the model performance. We will just try a simple grid and compare the results on testing dataset directly. (You need to further split the training dataset in cross validation to find the optimal hyper parameters, to prevent from "label leak"). Here the best hyper parameters the demo found is (1, 900, 1, 10), which achieves a precision at 10 of 18.8% in testing dataset, which means among the top 10 recommended items the user never bought in training data, on average he will buy 1.88 items on testing dataset, we had a very considerate recommender!)

# Benchmarking performance

To have a better idea of 




# Recommendation usage

