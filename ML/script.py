from time import time
from IPython.display import display
import pandas as pd
import numpy as np
from numpy import mean
from numpy import absolute
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sb
from scipy.stats.mstats import winsorize
import pickle 
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import (
    LinearRegression,
    TheilSenRegressor,
    RANSACRegressor,
    HuberRegressor,
)
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', None)

# Read the data
news_df = pd.read_csv("Dataset/OnlineNews.csv")

# Data Cleaning

# remove space character in column name
news_df.columns = news_df.columns.str.replace(' ', '')

# Removing useless features (url and timedelta) and list of all the useful features (descriptive and target features)
news_df.drop("url", axis='columns', inplace=True)
news_df.drop("timedelta", axis='columns', inplace=True)

# Identifying outliers and remove them
# Making a copy of the dataset to check the existence of outliers 
# and evaluate the effect of removing them before final outliers removal
c_news_df = news_df.copy()

#Function to create a list of features with outliers
def found_columns_with_outliers(data):
    columns = []
    for column in data.columns:
        #Apply for features with 3 or more unique values (Does not include binary features)
        if news_df[column].nunique() > 2:  
            for value in data[column]:
                if value:
                    columns.append(column)
                    break
    return columns


def get_outliers(df):
    #Using inter-quartile range to detect outliers
    q25, q75 = df.quantile(0.25), df.quantile(0.75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    outliers = (df < lower) | (df > upper)

    # Getting the list of features with outliers
    columns_with_outliers = found_columns_with_outliers(outliers)
        
    #Printing Results without target feature
    print('The number of Columns with outliers: {}'.format(len(columns_with_outliers)-1))
    print(columns_with_outliers[:-1])

    return columns_with_outliers



# Based on our observation, 3 features have only one outlier. we checked the index of the outlier for each feature
# the oulier in 3 features is the same sample (row) in our dataset. So, we will remove that sample or row.

indices = c_news_df[c_news_df['n_unique_tokens'] > 200].index[0]
c_news_df = c_news_df.drop(index=indices)

transformation = {'n_tokens_title':4, 'n_tokens_content':2, 'n_unique_tokens':4, 'n_non_stop_words':3, 'n_non_stop_unique_tokens':4, 
 'num_hrefs':2, 'num_self_hrefs':2, 'num_imgs':1, 'num_videos':2, 'average_token_length':4, 
 'num_keywords':3, 'kw_min_min':2, 'kw_max_min':2, 'kw_avg_min':2, 'kw_min_max':2, 'kw_max_max':3,
 'kw_avg_max':2, 'kw_max_avg':2, 'kw_avg_avg':4, 'self_reference_min_shares':2, 'self_reference_max_shares':2, 
 'self_reference_avg_sharess':2, 'LDA_00':1, 'LDA_01':1, 'LDA_02':1, 'LDA_03':1, 'global_subjectivity':4, 
 'global_sentiment_polarity':4, 'global_rate_positive_words':2, 'global_rate_negative_words':2, 
 'rate_positive_words':3, 'rate_negative_words':2, 'avg_positive_polarity':4, 'min_positive_polarity':2, 
 'avg_negative_polarity':3, 'max_negative_polarity':3, 'title_sentiment_polarity':4, 'abs_title_sentiment_polarity':2}

for k in transformation:
    if transformation[k] == 1:
        c_news_df[k] = np.log(c_news_df[k])
        
    elif transformation[k] == 2:
        c_news_df[k] = winsorize(c_news_df[k],(0,0.10))

    elif transformation[k] == 3:
        c_news_df[k] = winsorize(c_news_df[k],(0.10,0))
        
    elif transformation[k] == 4:
        c_news_df[k] = winsorize(c_news_df[k],(0.10,0.10))


# Based on our observation, 3 features have only one outlier. we checked the index of the outlier for each feature
# the oulier for them is the same sample (row) in our dataset. So, we will remove that sample or row.
indices = news_df[news_df['n_unique_tokens'] > 200].index[0]
news_df = news_df.drop(index=indices)

transformation = {'n_tokens_title':4, 'n_tokens_content':2, 'n_unique_tokens':4, 'n_non_stop_words':4, 'n_non_stop_unique_tokens':4, 
 'num_hrefs':2, 'num_self_hrefs':2, 'num_videos':2, 'average_token_length':4, 'num_keywords':3, 'kw_max_min':2, 'kw_avg_min':2, 'kw_avg_max':2, 'kw_max_avg':2, 'kw_avg_avg':4, 'global_subjectivity':4, 
 'global_sentiment_polarity':4, 'global_rate_positive_words':2, 'global_rate_negative_words':2, 
 'rate_positive_words':3, 'rate_negative_words':2, 'avg_positive_polarity':4, 'min_positive_polarity':2, 
 'avg_negative_polarity':3, 'max_negative_polarity':3, 'abs_title_sentiment_polarity':2}

for k in transformation:
    if transformation[k] == 2:
        news_df[k] = winsorize(news_df[k],(0,0.10))

    elif transformation[k] == 3:
        news_df[k] = winsorize(news_df[k],(0.10,0))
        
    elif transformation[k] == 4:
        news_df[k] = winsorize(news_df[k],(0.10,0.10))

news_df_describe = news_df.describe().T

# Separate X and Y
X = news_df.drop(['shares'], axis=1)
y = news_df['shares']



# scale data
cols = X.columns
scaler = RobustScaler()
scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(scaled, columns=cols)

# Build baseline models


# create split
# X_train - 80%(Input), X_test -> 20% input, Y_train -> (80%) output column, y_test -> 20% output column
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


# pipe = []
# pipe.append(('GB',Pipeline([('GBR', GradientBoostingRegressor(loss='absolute_error'))])))


# num_folds = 10
# seed = 111
# scoring = ['neg_mean_absolute_error', 'neg_root_mean_squared_error']
# scoring_for_print = {'fit_time':'fit_time', 'score_time': 'score_time', 'test_neg_mean_absolute_error':'test_MAE', 'train_neg_mean_absolute_error':'train_MAE',
#                      'test_neg_root_mean_squared_error':'test_RMSE', 'train_neg_root_mean_squared_error':'train_RMSE'
#                      }


# def build_models(pipe, X_train, y_train):
#     results = {}
#     names = {}
#     for name, regressor in pipe:
#         kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
#         reg_results = cross_validate(regressor, X_train, y_train, cv=kfold, scoring=scoring, return_train_score=True)
#         print(f'{name} completed')
#         res = {}
#         for key, values in reg_results.items():
#             res[scoring_for_print[key]] = np.mean(np.absolute(values))
#         results[name] = res
#     res_df = pd.DataFrame.from_dict(results, orient='index')
#     return res_df


# # feature selection for f_regression and mutual information
# def select_features(X_train, y_train, X_test, score_function, selectk):

#     fs = SelectKBest(score_func=score_function, k=selectk)

#     fs.fit(X_train, y_train)
#     X_train_fs = fs.transform(X_train)

#     X_test_fs = fs.transform(X_test)
    
#     mask = fs.get_support()
#     selectedFeatures = []

#     for bool, feature in zip(mask, X_train.columns.values):
#         if bool:
#             selectedFeatures.append(feature)

#     return X_train_fs, X_test_fs, fs, selectedFeatures

# def feature_selection_GridSearch(score_function, X_train, y_train):
#     # define the evaluation method
#     cv = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

#     # define the pipeline to evaluate
#     model = LinearRegression()
#     fs = SelectKBest(score_func=score_function)
#     pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])

#     # define the grid
#     grid = dict()
#     grid['sel__k'] = [i for i in range(X_train.shape[1]-50, X_train.shape[1]+1)]

#     # define the grid search
#     search = GridSearchCV(pipeline, grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)

#     # perform the search
#     results = search.fit(X_train, y_train)
    
#     # summarize best
#     print('Best MAE: %.3f' % results.best_score_)
#     print('Best Config: %s' % results.best_params_)

#     # summarize all
#     means = results.cv_results_['mean_test_score']
#     params = results.cv_results_['params']
#     for mean, param in zip(means, params):
#         print('>%.3f with: %r' % (mean, param))
        
#     return results.best_params_['sel__k'], abs(results.best_score_)


# def Train_Test_Split_build_models(pipe, X, y):
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
#     results = {}
#     names = {}
    
#     for name, regressor in pipe:
        
#         res = {}
        
#         start = time()
#         regressor.fit(X_train, y_train)
#         end = time()
#         res['fit_time'] = "{:.3f}".format(end - start)

#         predictions_train = regressor.predict(X_train)
#         start = time()
#         predictions_test = regressor.predict(X_test)
#         end = time()
#         res['score_time'] = "{:.3f}".format(end - start)

#         res['test_MAE'] = "{:.3f}".format(mean_absolute_error(y_test, predictions_test))
#         res['train_MAE'] = "{:.3f}".format(mean_absolute_error(y_train, predictions_train))
        
#         res['test_RMSE'] = "{:.3f}".format(mean_squared_error(y_test, predictions_test, squared=False))
#         res['train_RMSE'] = "{:.3f}".format(mean_squared_error(y_train, predictions_train, squared=False))
        
        
#         results[name] = res
        
#         print(f'{name} completed')
        
#     res_df = pd.DataFrame.from_dict(results, orient='index')
#     return res_df


# # Cross-validation
# base_df = build_models(pipe, X_train, y_train)


final_based_RG = GradientBoostingRegressor(loss='ls')
final_based_RG.fit(X_train, y_train)


y_val_prediction = final_based_RG.predict(X_val)


y_test_prediction = final_based_RG.predict(X_test)

y_test_arr = list(np.array(y_val))
y_test_pred_arr = list(np.array(y_val_prediction))

#print(np.array(y_test_prediction));
# print(type(y_test_arr))
# print(type(y_test_pred_arr))
for i in range(len(y_test_arr)):
    print(str(y_test_arr[i]) + "  " + str(y_test_pred_arr[i]))

print('\n\n')
MAE = mean_absolute_error(y_test, y_test_prediction)
MRSE = mean_squared_error(y_test, y_test_prediction, squared=False)
print(f'Mean Absolute Error: %.3f' % MAE)
print(f'Root Mean Square Error: %.3f' % MRSE)