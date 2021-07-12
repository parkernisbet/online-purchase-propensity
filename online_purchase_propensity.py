# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [md]
# # Online Purchase Propensity

# %% [md]
# ## Project Overview

# %% [md]
# This [Kaggle](https://www.kaggle.com/benpowis/customer-propensity-to-purchase-data) dataset 
# tracks a day's worth of user visits for a ficticious online retail website. 607055 rows, 25 
# columns, and (excluding "UserID") entirely composed of binary data. Features #2 through #24 track 
# various user states and actions across the website, while #25 denotes whether the user session 
# resulted in an order. The end goal is to prototype a couple machine learning models to predict 
# user propensity to purchase for our hypothetical client.

# %% [md]
# ## Data Exploration

# %%
# module imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from hyperopt import fmin, hp, STATUS_OK, tpe, Trials
from imblearn.ensemble import BalancedRandomForestClassifier
from IPython.display import Markdown, display
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %%
# markdown printer
def printmd(text):
    display(Markdown(text))

# %%
# importing data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
printmd('**Train info:**')
print(train.info(), '\n')
printmd('**Test info:**')
print(test.info())

# %% [md]
# Both dataframes are quite clean, no missing or null values. Dropping the "UserID" column as it 
# won't be needed in the remaining analysis.

# %%
# dropping userid
train.drop("UserID", axis = 1, inplace = True)
test.drop("UserID", axis = 1, inplace = True)

# %%
# class balances
printmd('**Train balance:**')
print(train.ordered.value_counts()/train.shape[0], '\n')
printmd('**Test balance:**')
print(test.ordered.value_counts()/test.shape[0])

# %% [md]
# This dataset suffers from heavy class imbalance, something we will need to keep in mind 
# when building predictive models sensitive to said imbalance. Note that the test dataset is 
# composed entirely of "0" "ordered" instances. This is expected to have a significant impact on 
# certain metrics for assessing model performance.

# %%
# correlation plot
fig, ax = plt.subplots()
fig.set_size_inches(10, 8)
sb.heatmap(train.corr(), cmap = 'winter', ax = ax)
ax.set_title('Column Correlation Matrix')
plt.show()

# %% [md]
# The above heatmap helps visualize predictor-predictor and predictor-response correlations at a 
# glance. "checked_delivery_detail", "sign_in", and "saw_checkout" all show promising amounts of 
# correlation with the response variable, albeit with similarly high levels of multicollinearity 
# between themselves.
#
# A more quantitative method of assessing multicollinearity amongst the predictors would be to 
# be to calculate variance inflation factors (VIFs).

# %%
# x-y splits
X_train = train.iloc[:, :-1]
y_train = train['ordered']
X_test = test.iloc[:, :-1]
y_test = test['ordered']

# %%
# calculating vifs
vifs = pd.DataFrame(X_train.columns, columns = ['col'])
vifs.loc[:, 'vif'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vifs = vifs.sort_values('vif', ascending = False).set_index('col', drop = True)
print(vifs)

# %% [md]
# The top four variables in this list make up those first to be addressed: "loc_uk", 
# "device_mobile", "sign_in", and "saw_checkout". In a much wider dataset, feature engineering and 
# reduction might be necessary to wrangle in our model; our current set of 23 features will be kept 
# as is.

# %% [md]
# ## Building Models

# %% [md]
# ### 1. BernoulliNB (bnb)

# %% [md]
# The first model will serve as the baseline to beat, a derivative of the naive bayes theorem 
# specifically for boolean features. For reasons covered later on, this model should perform well 
# despite some hidden gotchas in this particular dataset.

# %%
# model construction
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
y_pred = bnb.predict(X_test)
printmd('**Classification Report**')
print(classification_report(y_test, y_pred, digits = 4, zero_division = 0))

# %% [md]
# The impact of only having one class in the test dataset is evident in the class "1" f1 score (0) 
# and macro average f1 score (.495). Both metrics suffer with no true positives for class "1"; 
# false positives make up the only "1" class predictions and therefore tank precision and recall. 
# For a better representation of predictive performance on a two-class test dataset, we turn to 
# stratified cross validation of the training dataset.

# %%
# validation function
def model_validation(clf, X, y, format = 'report'):
    '''
    Stratified cross validation function, outputs either classification_report or f1_score.

        Arguments:
            clf (object): classifier, supports .fit / .predict
            X (array): feature array, n x d
            y (array): label array, n x 1
            format (string): output format, 'report' or 'score'
        
        Returns:
            result (string / float): validation metric, depends on format variable
    '''
    preds = []
    tests = []
    skf = StratifiedKFold(shuffle = True)
    for train_ind, test_ind in skf.split(X, y):
        train_X, test_X = X.values[train_ind], X.values[test_ind]
        train_y, test_y = y[train_ind], y[test_ind]
        clf.fit(train_X, train_y)
        pred_y = clf.predict(test_X)
        preds.append(pred_y)
        tests.append(test_y)
    if format == 'report':
        result = classification_report(np.hstack(tests), np.hstack(preds), digits = 4)
    else:
        result = f1_score(np.hstack(tests), np.hstack(preds), average = 'macro')
    return result

# %%
# bernoulli validation
printmd('**Classification Report**')
bnb = BernoulliNB()
print(model_validation(bnb, X_train, y_train))

# %% [md]
# Now that the naive bayes model is predicting on instances from both classes, we get a more 
# realistic look at predictive performance. Our cross validated model's precision for class "1" 
# does leave something to be desired, something we hope to improve with another model. 
#
# Above it was mentioned that the heavy class imbalance would impact models sensitive to said 
# imbalance. BernoulliNB is a probabilistic, binary classification model (in this specific 
# instance), and therefore is mostly unaffected. Any deterministic models we attempt to implement 
# will need to take steps to address this imbalance.

# %% [md]
# ### 2. BalancedRandomForestClassifier (brf)
#
# Enter "imbalanced-learn" (also "imblearn"), a Python library built for the explicit purpose 
# of dealing with class imbalanced datasets. Touted for it's high degree of compatibility with 
# scikit-learn, "imbalanced-learn" contains a hodge podge of resampling and 
# sklearn-wrapped methods tailor-built for handling unbalanced datasets.
#
# In contrast to sklearn's RandomForestClassifier, brf includes parameters to control random 
# undersampling. Each weak learner in the bagged model is trained on an undersampled (with 
# replacement) subset of the original dataset, albeit with a more favorable / equal balance of 
# classes.

# %%
# parameter space
params = {
    'n_estimators': hp.quniform('n_estimators', 20, 80, 1),
    'max_depth': hp.quniform('max_depth', 1, 15, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 50, 1),
    'max_features': hp.quniform('max_features', 1, 23, 1),
    'sampling_strategy': hp.choice(
        'sampling_strategy',
        ['majority', 'not minority', 'not majority', 'all']
    ),
    'replacement': hp.choice('replacement', ['True', 'False']),
    'class_weight': hp.choice('class_weight', [None, 'balanced', 'balanced_subsample'])
}

# %%
# optimization function
def objective(params):
    keys = list(params.keys())
    floats = ['n_estimators', 'max_depth', 'min_samples_split', 'max_features']
    params = {k:(int(params[k]) if k in floats else params[k]) for k in keys}
    brf = BalancedRandomForestClassifier(**params, n_jobs = -1, random_state = 42)
    score = model_validation(brf, X_train, y_train, 'score')
    return {'loss': -score, 'status': STATUS_OK}

# %% [md]
# To eke out as much predictive performance as possible, we will be relying on HyperOpt for some
# hyperparameter tuning. The fmin function below will test various combinations of 
# hyperparameters from a pre-defined parameter space, using outcomes to guide it to the optimal 
# set. Compared to GridSearch and RandomSearch, this translates to less time spent testing 
# conditions with low chances of improving performance.
#
# One quick warning, this notebook was run on a 24 core system and the below cell took 12 minutes 
# and 57 seconds to run. Your execution time will scale based on how your system compares with mine.

# %%
# tuning model
trials = Trials()
best = fmin(fn = objective, space = params, algo = tpe.suggest, max_evals = 50, trials = trials)
sampling_strategy = ['majority', 'not minority', 'not majority', 'all']
replacement = ['True', 'False']
class_weight = [None, 'balanced', 'balanced_subsample']
for name in ['sampling_strategy', 'replacement', 'class_weight']:
    best[name] = eval(f'{name}[best[name]]')
for name in ['n_estimators', 'max_depth', 'min_samples_split', 'max_features']:
    best[name] = int(best[name])
printmd('**Best Parameters**')
print(best, '\n')
printmd('**Best Macro F1 Score**')
print(-trials.best_trial['result']['loss'])

# %% [md]
# This second model and macro f1 score mark an improvement of ~.055 over the previous. Some notes 
# about the best parameter set from our hyperparameter tuning:
#
#   1. the chosen combination of parameters varies quite wildly
#   2. most stable parameters are "n_estimators", "max_depth", and "min_samples_split"
#           a. "n_estimators" is usually between 40 and 60
#           b. "max_depth" is usually below 10
#           c. "min_samples_split" is usually between 15 and 25
#   3. macro f1 score achieved is usually in the .959 to .960 range
#
# Let's generate a classification report so we can compare per class f1 scores.

# %%
# model report
brf = BalancedRandomForestClassifier(**best, n_jobs = -1, random_state = 42)
printmd('**Classification Report**')
print(model_validation(brf, X_train, y_train))

# %% [md]
# ## Closing Thoughts

# %% [md]
# Swapping models resulted in f1 score net-positive changes for both classes, though only minimally 
# so for class "0". The gains for class "1" were in precision, meaning more correct "1" predictions 
# per overall "1" predictions; recall did suffer, but only minimally so. In a business context this 
# would means fewer false positives and more false negatives, respectively, but in general better 
# predictive abilities for class "1". Depending on client requirements these specific models could 
# be re-weighted, though the end result would be a trade off between each class' predictive power.
#
# Further exploration, post-project, would be to investigate more computationally complex models; 
# XGBoost, LightGBM, and Catboost come to mind.
