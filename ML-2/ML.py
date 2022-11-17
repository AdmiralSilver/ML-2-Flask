from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from joblib import dump

# 1**Frame the problem and look at the big picture**
# Unsure of we need to write anything about this or if all is written in the report
# 2**Get the data**
# Importing the dataset
datasetTrain = pd.read_csv('train.csv')
datasetTest = pd.read_csv('test.csv')
# 3**Explore and visualize the data to gain insights**
datasetTrain.info()
datasetTest.info()
print('-----------------')
print(datasetTest.head())
print('-----------------')
print(datasetTrain.head())
print('-----------------')
print(datasetTrain.describe())
print('-----------------')
print(datasetTest.describe())
# See missing values
print('Missing values-----------------')
print(datasetTrain.isnull().sum())
print('-----------------')
print(datasetTest.isnull().sum())

# Drop belongs_to_collection because most of the values are null
datasetTrain = datasetTrain.drop(['belongs_to_collection'], axis=1)
datasetTest = datasetTest.drop(['belongs_to_collection'], axis=1)
# Drop homepage because most of the values are null
datasetTrain = datasetTrain.drop(['homepage'], axis=1)
datasetTest = datasetTest.drop(['homepage'], axis=1)
print(datasetTest[datasetTest['release_date'].isnull()])
# The release date of the movie with id 3829 is missing, after an internet search we found that the release date is
# 05/01/2000
# We replace the missing value with the correct one
datasetTest.loc[datasetTest['release_date'].isnull(), 'release_date'] = '05/01/00'
print(datasetTest[datasetTest["release_date"] == '5/1/00'])
# For nominal data (strings), we replace the missing values with "unknown"
datasetTrain[['genres',
              'original_language',
              'production_companies',
              'production_countries',
              'status',
              'cast',
              'crew',
              'spoken_languages',
              ]] = datasetTrain[['genres',
                                 'original_language',
                                 'production_companies',
                                 'production_countries',
                                 'status',
                                 'cast',
                                 'crew',
                                 'spoken_languages',
                                 ]].fillna('unknown')
datasetTest[['genres',
             'original_language',
             'production_companies',
             'production_countries',
             'status',
             'cast',
             'crew',
             'spoken_languages',
             ]] = datasetTest[['genres',
                               'original_language',
                               'production_companies',
                               'production_countries',
                               'status',
                               'cast',
                               'crew',
                               'spoken_languages',
                               ]].fillna('unknown')
# For numerical data, we replace the missing values with the mean
datasetTrain['runtime'] = datasetTrain['runtime'].fillna(datasetTrain['runtime'].mean())
datasetTest['runtime'] = datasetTest['runtime'].fillna(datasetTest['runtime'].mean())
print(datasetTrain['runtime'].isnull().any())
print(datasetTest['runtime'].isnull().any())
# We check if there are still missing values
print('Missing values-----------------')
print(datasetTrain.isnull().sum())
print('-----------------')
print(datasetTest.isnull().sum())
# We need to convert the release_date column to datetime
datasetTrain['release_date'] = pd.to_datetime(datasetTrain['release_date'])
datasetTest['release_date'] = pd.to_datetime(datasetTest['release_date'])
# We create a new column with the year, month and day of the release date
datasetTrain['release_year'] = pd.to_datetime(datasetTrain['release_date']).dt.year.astype(int)
datasetTrain['release_month'] = pd.to_datetime(datasetTrain['release_date']).dt.month.astype(int)
datasetTrain['release_day'] = pd.to_datetime(datasetTrain['release_date']).dt.day.astype(int)
datasetTest['release_year'] = pd.to_datetime(datasetTest['release_date']).dt.year.astype(int)
datasetTest['release_month'] = pd.to_datetime(datasetTest['release_date']).dt.month.astype(int)
datasetTest['release_day'] = pd.to_datetime(datasetTest['release_date']).dt.day.astype(int)
# We drop the release_date column since we don't need it anymore
datasetTrain = datasetTrain.drop(['release_date'], axis=1)
datasetTest = datasetTest.drop(['release_date'], axis=1)
# Considering the competition was in 2019, there should not be any movies with a release date after 2019
print("Maximum release_year in train-set: ", datasetTrain['release_year'].max())
print("Maximum release_year in test-set: ", datasetTest['release_year'].max())


# We can see that quite a few movies have a release year after 2019, seems like these movies were released in the
# 1900s but a mistake has swapped 19 with 20 so that a movie released in 1971 is registered as 2071
# Fixing the release year
def fix_release_year(year):
    if year > 2019:
        return year - 100
    else:
        return year


datasetTrain['release_year'] = datasetTrain['release_year'].apply(lambda x: fix_release_year(x))
datasetTest['release_year'] = datasetTest['release_year'].apply(lambda x: fix_release_year(x))

# **Analyzing the data**
# Visualizing the budget
sns.set(rc={'figure.figsize': (15, 8)})
plt.xlabel('Budget')
plt.hist(datasetTrain['budget'], bins=50)
plt.show()
# From the plot we can see that most of the movies have low budget
# Display the relation between budget and revenue
sns.set(rc={'figure.figsize': (15, 8)})
plt.xlabel('Budget')
plt.ylabel('Revenue')
plt.scatter(datasetTrain['budget'], datasetTrain['revenue'])
plt.show()
# From the plot we can see that there is a positive correlation between budget and revenue
# Display the relation between budget and popularity
sns.set(rc={'figure.figsize': (15, 8)})
plt.xlabel('Budget')
plt.ylabel('Popularity')
plt.scatter(datasetTrain['budget'], datasetTrain['popularity'])
plt.show()
# Correlation matrix
corrMatrix = datasetTrain.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()
#  From the correlation matrix we can see that there is a high positive
#  correlation between budget and revenue, there is also a high positive correlation between popularity and revenue,
#  lastly there is a positive correlation between runtime and revenue
#  Visualizing these correlations
fig, ax = plt.subplots(2, 3, figsize=(15, 8), tight_layout=True)

datasetTrain.plot(ax=ax[0][0], x='budget', y='revenue', style='o', ylabel='Revenue', color='red').set_title(
    'Budget vs Revenue')
datasetTrain.plot(ax=ax[0][1], x='popularity', y='revenue', style='o', ylabel='Revenue', color='green').set_title(
    'Popularity vs Revenue')
datasetTrain.plot(ax=ax[0][2], x='runtime', y='revenue', style='o', ylabel='Revenue', color='blue').set_title(
    'Runtime vs Revenue')
datasetTrain.plot(ax=ax[1][0], x='budget', y='popularity', style='o', ylabel='Popularity', color='orange').set_title(
    'Budget vs Popularity')
datasetTrain.plot(ax=ax[1][1], x='budget', y='runtime', style='o', ylabel='Runtime', color='purple').set_title(
    'Budget vs Runtime')
datasetTrain.plot(ax=ax[1][2], x='popularity', y='runtime', style='o', ylabel='Runtime', color='brown').set_title(
    'Popularity vs Runtime')
plt.show()

# Visualizing the change in revenue, runtime, popularity and budget over the years
fig, ax = plt.subplots(4, 1, tight_layout=True)
plt.grid()

datasetTrain.groupby('release_year')['revenue'].mean().plot(ax=ax[0], figsize=(10, 10), linewidth=3,
                                                            color='red').set_title('Revenue over the years')
datasetTrain.groupby('release_year')['runtime'].mean().plot(ax=ax[1], figsize=(10, 10), linewidth=3,
                                                            color='green').set_title('Runtime over the years')
datasetTrain.groupby('release_year')['popularity'].mean().plot(ax=ax[2], figsize=(10, 10), linewidth=3,
                                                               color='blue').set_title('Popularity over the years')
datasetTrain.groupby('release_year')['budget'].mean().plot(ax=ax[3], figsize=(10, 10), linewidth=3,
                                                           color='orange').set_title('Budget over the years')
plt.show()
# TODO - Comment on the plots and their correlations
print("Movies with budget under 10000: ", len(datasetTrain[datasetTrain['budget'] < 10000]))
# We can see that there are 835 out of 300 movies that have a budget under 10000, TODO - do something with these?
# TODO cont- Done later in the code
#  TODO - maybe some more exploration and visualization
#  4**Prepare the data for Machine Learning algorithms**
#   TODO - Prepare the data for Machine Learning algorithms TODO - Genres, cast, spoken_languages and crew in JSON
#    TODO cont -format, convert to nominal values.  These columns could be used to create new features such as the
#     number of genres, number of cast members, number of spoken languages, number of crew members, etc.
# Many of the features that could be useful is in JSON-format, for example the genres column
for y in enumerate(datasetTest['genres'][:10]):
    print(y)


# Converting JSON to nominal format
def convert_data(x):
    try:
        data = eval(x)
    except:
        data = {}
    return data


datasetTrain.genres = datasetTrain.genres.map(lambda x: sorted([i['name'] for i in convert_data(x)])).map(
    lambda x: ','.join(map(str, x)))
datasetTrain.spoken_languages = datasetTrain.spoken_languages.map(
    lambda x: sorted([i['name'] for i in convert_data(x)])).map(
    lambda x: ','.join(map(str, x)))
datasetTrain.crew = datasetTrain.crew.map(lambda x: sorted([i['name'] for i in convert_data(x)])).map(
    lambda x: ','.join(map(str, x)))
datasetTrain.cast = datasetTrain.cast.map(lambda x: sorted([i['name'] for i in convert_data(x)])).map(
    lambda x: ','.join(map(str, x)))

datasetTest.genres = datasetTest.genres.map(lambda x: sorted([i['name'] for i in convert_data(x)])).map(
    lambda x: ','.join(map(str, x)))
datasetTest.spoken_languages = datasetTest.spoken_languages.map(
    lambda x: sorted([i['name'] for i in convert_data(x)])).map(
    lambda x: ','.join(map(str, x)))
datasetTest.crew = datasetTest.crew.map(lambda x: sorted([i['name'] for i in convert_data(x)])).map(
    lambda x: ','.join(map(str, x)))
datasetTest.cast = datasetTest.cast.map(lambda x: sorted([i['name'] for i in convert_data(x)])).map(
    lambda x: ','.join(map(str, x)))

print(datasetTrain.crew.head())
# This is a better way to visualize the data compared to the json format
# But it still might be more interesting to see the amount of genres, cast members, spoken languages and crew members
# to see if there is a correlation between these and the revenue
# One could for example expect that a bigger crew would mean higher revenue

datasetTrain['genres_amount'] = datasetTrain['genres'].str.count(',') + 1
datasetTrain['cast_amount'] = datasetTrain['cast'].str.count(',') + 1
datasetTrain['spoken_languages_amount'] = datasetTrain['spoken_languages'].str.count(',') + 1
datasetTrain['crew_amount'] = datasetTrain['crew'].str.count(',') + 1

datasetTest['genres_amount'] = datasetTest['genres'].str.count(',') + 1
datasetTest['cast_amount'] = datasetTest['cast'].str.count(',') + 1
datasetTest['spoken_languages_amount'] = datasetTest['spoken_languages'].str.count(',') + 1
datasetTest['crew_amount'] = datasetTest['crew'].str.count(',') + 1
print(datasetTest['genres_amount'])

# Converting the nominal values to numerical values
# TODO - One hot encoding
datasetTrain[['status',
              'original_language',
              'production_companies',
              'production_countries']] = datasetTrain[['status',
                                                       'original_language',
                                                       'production_companies',
                                                       'production_countries']].astype('category')
datasetTrain['status'] = datasetTrain['status'].cat.codes
datasetTrain['original_language'] = datasetTrain['original_language'].cat.codes
datasetTrain['production_companies'] = datasetTrain['production_companies'].cat.codes
datasetTrain['production_countries'] = datasetTrain['production_countries'].cat.codes

datasetTest[['status',
             'original_language',
             'production_companies',
             'production_countries']] = datasetTest[['status',
                                                     'original_language',
                                                     'production_companies',
                                                     'production_countries']].astype('category')

datasetTest['status'] = datasetTest['status'].cat.codes
datasetTest['original_language'] = datasetTest['original_language'].cat.codes
datasetTest['production_companies'] = datasetTest['production_companies'].cat.codes
datasetTest['production_countries'] = datasetTest['production_countries'].cat.codes

print(datasetTrain['production_countries'])
# print out number of movies with budget of 0
print("Movies with budget of 0: ", len(datasetTrain[datasetTrain['budget'] == 0]))
# print out number of movies with runtime of 0
print("Movies with runtime of 0:", len(datasetTrain[datasetTrain['runtime'] == 0]))
# We can see that a lot of movies has a budget of 0, and some of these should be high budget movies
# It also makes no sense to have a runtime of 0
# We will replace the 0 values with the mean of the column
datasetTrain['budget'] = datasetTrain['budget'].replace(0, datasetTrain['budget'].mean())
datasetTrain['runtime'] = datasetTrain['runtime'].replace(0, datasetTrain['runtime'].mean())

datasetTest['budget'] = datasetTest['budget'].replace(0, datasetTest['budget'].mean())
datasetTest['runtime'] = datasetTest['runtime'].replace(0, datasetTest['runtime'].mean())
# New correlation matrix with the new features
corrMatrix = datasetTrain.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()
# From this new correlation matrix we can see that some new features are correlated with the revenue
# We chose to use the following features for our model: budget, popularity, runtime, cast_amount, crew_amount
# We will now try to predict the revenue using these features
# Predictor variables
X = datasetTrain[['budget', 'popularity', 'runtime', 'cast_amount', 'crew_amount']]
# Target variable
y = datasetTrain['revenue']
# Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# 5**Explore many different models and shortlist the best ones**
# We will try out the following models:
# Linear Regression
# Decision Tree
# Random Forest
# Support Vector Machine
# K Nearest Neighbors
# Gradient Boosting
# We will use the mean squared error as a metric to evaluate the models
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("Linear Regression MSE: ", mean_squared_error(y_test, y_pred))
# Decision Tree
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print("Decision Tree MSE: ", mean_squared_error(y_test, y_pred))
# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Random Forest MSE: ", mean_squared_error(y_test, y_pred))
# Support Vector Machine
svr = SVR()
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
print("Support Vector Machine MSE: ", mean_squared_error(y_test, y_pred))
# K Nearest Neighbors
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("K Nearest Neighbors MSE: ", mean_squared_error(y_test, y_pred))
# Gradient Boosting
gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
print("Gradient Boosting MSE: ", mean_squared_error(y_test, y_pred))
# Scores
# Linear Regression MSE:        7950571505110897.0
# Decision Tree MSE:            1.2655086106528986e+16
# Random Forest MSE:            6706835613617110.0
# Support Vector Machine MSE:   2.2173175065004816e+16
# K Nearest Neighbors MSE:      9751561958431006.0
# Gradient Boosting MSE:        7184735606350928.0
# We can see that the Random Forest model performed the best

# 6**Fine-tune your models and combine them into a great solution**
# TODO - Fine-tune your models and combine them into a great solution
# We will now try to improve the model by tuning the hyperparameters
# We will use RandomizedSearchCV to find the best hyperparameters
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
# Random search of parameters, using 3-fold cross validation,
# search across 100 different combinations, and use all available cores
# <-- rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
# <--                              random_state=42, n_jobs=-1)
# Fit the random search model
# <-- rf_random.fit(X_train, y_train)
# Best parameters
# <-- print("Best parameters: ", rf_random.best_params_)
# Best score
# <-- print("Best score: ", rf_random.best_score_)
# Best parameters {'n_estimators': 2000, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt',
# 'max_depth': 10, 'bootstrap': False} Best score 0.688
# We will now use the best parameters to train the model
rf = RandomForestRegressor(n_estimators=400, min_samples_split=10, min_samples_leaf=4, max_features='auto',
                           max_depth=70, bootstrap=True)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Random Forest MSE: ", mean_squared_error(y_test, y_pred))
# Submission
submission = pd.DataFrame({'Id': datasetTest['id'], 'revenue': rf.predict(
    datasetTest[['budget', 'popularity', 'runtime', 'cast_amount', 'crew_amount']])})
submission.to_csv('submission.csv', index=False)
print(submission.head())
# Compare the results of the model with the actual revenue with a bar chart
barData = {'Actual revenue': y_test.mean(), 'Predicted revenue': y_pred.mean()}
barOne = list(barData.keys())
barTwo = list(barData.values())

fig = plt.figure(figsize=(10, 5))
plt.bar(barOne, barTwo, color=['blue', 'orange'])
plt.xlabel("Revenue")
plt.ylabel("Amount")
plt.title("Actual vs Predicted revenue")
plt.show()
# Feature importance
# We will now try to find out which features are the most important
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index=X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)
# From the results we can see that the most important features are budget and popularity What we learn from this is
# that the budget and popularity of a movie are the most important factors in determining the revenue of a movie We
# can also see that the runtime, cast amount and crew amount are not very important Going forward we can try to
# remove these features and see if the model performs better
NB_DIR = Path.cwd()
MODEL_DIR = NB_DIR/'models'


dump(rf, MODEL_DIR/'model.joblib', compress=6)
# 7**Present your solution**
# TODO - Present your solution
# 8**Launch, monitor, and maintain your system**
# TODO - Launch, monitor, and maintain your system
