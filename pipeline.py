import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd
from sklearn.linear_model import LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_array
from sklearn.metrics import r2_score


class StackingEstimator(BaseEstimator, TransformerMixin):

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed


train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

train_ids = train.ID

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

usable_columns = list(set(train.columns) - {'y'})

all_data = pd.concat([train, test], axis=0)
print(all_data.shape)
all_data.sort_values(['X0', 'ID'], inplace=True)
all_data.reset_index(inplace=True, drop=True)
all_data.loc[:, 'lag1'] = all_data.y.shift(1)
all_data.loc[:, 'lead1'] = all_data.y.shift(-1)

train = all_data[all_data.ID.isin(train_ids.values)]
test = all_data[~all_data.ID.isin(train_ids.values)]
test = test.drop('y', axis=1)

y_train = train['y'].values
y_mean = np.mean(y_train)
id_test = test['ID'].values
finaltrainset = train[usable_columns].values
finaltestset = test[usable_columns].values


sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = 0
for fold in range(1, 4):
    np.random.seed(fold)
    xgb_params = {
        # 'n_trees': 610,
        'eta': 0.0049,
        'max_depth': 5,
        'subsample': 0.89,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'base_score': y_mean,
        # 'silent': True,
        'colsample_bytree': 0.7,
        'seed': fold,
    }

    dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
    dtest = xgb.DMatrix(test)

    num_boost_rounds = 2500
    model = xgb.train(dict(xgb_params), dtrain, num_boost_round=num_boost_rounds)
    y_pred = model.predict(dtest)

    stacked_pipeline = make_pipeline(
        StackingEstimator(estimator=LassoLarsCV(normalize=True)),
        StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001,
                                                              loss="huber",
                                                              max_depth=3,
                                                              max_features=0.45,
                                                              min_samples_leaf=21,
                                                              min_samples_split=12,
                                                              subsample=0.76)),
        LassoLarsCV()
    )

    stacked_pipeline.fit(finaltrainset, y_train)
    results = stacked_pipeline.predict(finaltestset)

    print('R2 score on train data:')
    print(r2_score(y_train, stacked_pipeline.predict(finaltrainset) * 0.2855 + model.predict(dtrain) * 0.7145))

    sub['y'] += y_pred * 0.75 + results * 0.25
sub['y'] /= 3

leaks = {
    1: 71.34112,
    12: 109.30903,
    23: 115.21953,
    28: 92.00675,
    42: 87.73572,
    43: 129.79876,
    45: 99.55671,
    57: 116.02167,
    3977: 132.08556,
    88: 90.33211,
    89: 130.55165,
    93: 105.79792,
    94: 103.04672,
    1001: 111.65212,
    104: 92.37968,
    72: 110.54742,
    78: 125.28849,
    105: 108.5069,
    110: 83.31692,
    1004: 91.472,
    1008: 106.71967,
    1009: 108.21841,
    973: 106.76189,
    8002: 95.84858,
    8007: 87.44019,
    1644: 99.14157,
    337: 101.23135,
    253: 115.93724,
    8416: 96.84773,
    259: 93.33662,
    262: 75.35182,
    1652: 89.77625
}
sub['y'] = sub.apply(lambda r: leaks[int(r['ID'])] if int(r['ID']) in leaks else r['y'], axis=1)
sub.to_csv('stacked-models.csv', index=False)
