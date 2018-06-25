#####################################################
# Date: June, 21, 2018                              #
# Author: Yang Jiang                                #
# Contact: yangjiang.us@gmail.com                   #
#                                                   #
# For EY internal communication only                #
#                                                   #
# COPYRIGHT © 2018 YANG JIANG ALL RIGHTS RESERVED   #
#####################################################

import implicit
import pandas as pd
import numpy as np
from scipy import sparse
import itertools
from datetime import datetime
import matplotlib.pyplot as plt
import random

# read in retail data, data can be downloaded from 'http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
# see 'http://archive.ics.uci.edu/ml/datasets/Online+Retail' for more details 
data = pd.read_excel('data/Online Retail.xlsx')


##### start data cleansing

# check variable type and missing value
print(data.head())
print(data.info())
print(data.describe())

# check no. of user and item before data cleansing
print('Number of users: ', data.CustomerID.nunique())
print('Number of items: ', data.Description.nunique())

# remove missing value and negative quantity/price
data.dropna(inplace=True)
data = data[data.Quantity>0]
data = data[data.UnitPrice>0]

# set customerID as int
data.CustomerID = data.CustomerID.astype(int)

# aggregate the total quantity of each item over all time
group = data.groupby(['CustomerID', 'Description'])
data = group['Quantity'].aggregate(np.sum).reset_index()

# remove customers with less than 10 item purchases
group = data.groupby('CustomerID')
data = data[data.CustomerID.isin(group.size()[group.size()>=10].index)]
data.reset_index(drop=True, inplace=True)

# rename variables
data.rename(columns={'CustomerID': 'User', 'Description': 'Item'}, inplace=True)

# check no. of user, item and sparsity after data cleansing
n_user = data.User.nunique()
n_item = data.Item.nunique()
print('Number of customers: ', n_user)
print('Number of items: ', n_item)
print('Sparsity: ', round(data.shape[0]/n_user/n_item*100, 4), "%")


##### start data preprocessing

# transform user and item into categorical variables and save a mapping list
user_map = pd.DataFrame({"UserID": range(n_user),
                         "User": data.User.astype("category").cat.categories})
item_map = pd.DataFrame({"ItemID": range(n_item),
                         "Item": data.Item.astype("category").cat.categories})
data['UserID'] = data.User.astype("category").cat.codes
data['ItemID'] = data.Item.astype("category").cat.codes

# randomly select 20% customer who buys 20+ items 
np.random.seed(0)
user_test_bi = pd.Series([0]*n_user)
total_item = data.groupby('UserID').size()
over20 = total_item.index[total_item>=20]
user_test = np.sort(np.random.choice(over20, round(0.2*len(over20)), replace=False))
user_test_bi[user_test] = 1

# for each selected customer, exclude 10 items to test dataset
current = 0
test_index = []
for i in range(n_user):
    if user_test_bi[i]==1:
        test_index += list(np.random.choice(range(current, current+total_item[i]), 10, replace=False))
    current += total_item[i]
test = data[['UserID', 'ItemID', 'Quantity']][data.index.isin(test_index)]
test.reset_index(drop=True, inplace=True)
train = data[['UserID', 'ItemID', 'Quantity']][~data.index.isin(test_index)]
train.reset_index(drop=True, inplace=True)


##### fitting recommendation algorithm

# the function implicit.als.AlternatingLeastSquares() strongly recommends disabling multithreading, I don't know why...
import os
os.environ['MKL_NUM_THREADS'] = '1'

# create a sparse matrix 
train_csr = sparse.csr_matrix((train.Quantity, (train.UserID, train.ItemID)))

# define function calculating precision at k
# the input prediction is a n(users in test data) by k(recommended item ID) matrix
def patk(prediction, test):
    precision = []
    for i in range(len(user_test)):
        precision.append(len(set(prediction[i, ]) & 
                             set(test.ItemID[test.UserID==user_test[i]])) / prediction.shape[1])
    return(np.mean(precision))

# cross validation to find optimal values for hyper parameters
# here we just demonstrate with a simple example, which has not optimized the parameters
alpha = [1, 3, 9]
factors = [100, 300, 900]
regularization = [0.01, 0.1, 1]
iterations = [5, 10, 20]
cvlist = []
for i in itertools.product(alpha, factors, regularization, iterations):
    cvlist.append(i)

cv_patk = []
for i in range(len(cvlist)):
    model = implicit.als.AlternatingLeastSquares(factors=cvlist[i][1], regularization=cvlist[i][2], iterations=cvlist[i][3])
    alpha = cvlist[i][0]
    model.fit(train_csr.T.tocsr()*alpha)
    prediction = np.zeros(shape=(len(user_test), 10))
    for j in range(len(user_test)):
        prediction[j, ] = [x for (x, y) in model.recommend(user_test[j], train_csr*alpha)]
    cv_patk.append(patk(prediction, test))
    print("iteration: ", i, " current time:", str(datetime.now()))

# check optimal patk and corresponding parameters
patk_RS = np.max(cv_patk)
print(patk_RS)
# 0.1880597014925373
best_para = cvlist[np.argmax(cv_patk)]
print(best_para)
# (1, 900, 1, 10)

# Decide not to include this in
# benchmark with RS using default parameter setting
#model = implicit.als.AlternatingLeastSquares()
#model.fit(train_csr.T.tocsr())
#prediction = np.zeros(shape=(len(user_test), 10))
#for i in range(len(user_test)):
#    prediction[i, ] = [x for (x, y) in model.recommend(user_test[i], train_csr)]
#patk_default = patk(prediction, test)
#print(patk_default)
# 0.14693200663349917

# benchmark with approach of popularity: recommend most popular items
popular_item = train.groupby('ItemID').size()
popular_item = popular_item.sort_values(ascending=False).index.values
prediction = np.zeros(shape=(len(user_test), 10))
for i in range(len(user_test)):
    prediction[i, ] = popular_item[~np.isin(popular_item, train.ItemID[train.UserID==user_test[i]].values)][:10]
patk_popular = patk(prediction, test)
print(patk_popular)
# 0.03913764510779436

# benchmark with random approach: randomly recommend 10 unpurchased item
# calculate pool of random recommendation for each user in test data
all_item = pd.Series(range(n_item))
random_pool = []
for i in range(len(user_test)):
    random_pool.append(list(all_item[~np.isin(all_item, train.ItemID[train.UserID==user_test[i]].values)]))

# calculate patk with 100 iterations to reduce variance
patk_random_list = []
for i in range(100):
    prediction = np.zeros(shape=(len(user_test), 10))
    for j in range(len(user_test)):
        prediction[j, ] = np.random.choice(random_pool[j], 10, replace=False)
    patk_random_list.append(patk(prediction, test))
patk_random = np.mean(patk_random_list)
print(patk_random)
# 0.002648424543946932

# make a barplot to compare model performance
plt.figure(figsize=(5,5))
plt.barh([2, 1, 0], [patk_RS, patk_popular, patk_random], align='center', alpha=0.5)
plt.yticks([2, 1, 0], ['Implicit', 'Popular', 'Random'])
plt.xlabel('Model performance in terms of Precision at 10')
plt.title('Benchmark model performance with alternative approaches') 
plt.show()


##### explore the prediction result
# For a random customer Yang, list his top transaction history in train data
np.random.seed(0)
random.seed(0)
Yang = 1001
Yang_train = train[train.UserID==Yang]
Yang_train = Yang_train.loc[Yang_train.Quantity.sort_values(ascending=False).index]
Yang_train = Yang_train.merge(item_map, on='ItemID', how='left')
print(Yang_train.head(10))

# Check the recommendation for Yang and what he really bought in test data
model = implicit.als.AlternatingLeastSquares(factors=best_para[1], regularization=best_para[2], iterations=best_para[3])
alpha = best_para[0]
model.fit(train_csr.T.tocsr()*alpha)
Yang_test = pd.DataFrame(model.recommend(Yang, train_csr*alpha), columns=['ItemID', 'Score'])
Yang_test = Yang_test.merge(item_map, on='ItemID', how='left')
Yang_test = Yang_test.merge(test[test.UserID==Yang], on='ItemID', how='left')
print(Yang_test.drop(columns=['UserID']))

# For a random item, show a list of similar items
item = 123
similar_item = pd.DataFrame(model.similar_items(item), columns=['ItemID', 'Score'])
similar_item = similar_item.merge(item_map, on='ItemID', how='left')
print(similar_item)

