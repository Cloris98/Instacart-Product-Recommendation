import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import keras
from keras.utils.vis_utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier

warnings.filterwarnings("ignore")

model_all_data = pd.read_csv('model_all_data_added.csv')

# deal with categorical data
model_all_data = pd.concat(
    [model_all_data.drop('product_department', axis=1),
     pd.get_dummies(model_all_data["product_department"],
                    prefix='product_department_')], axis=1)

# split data
test_user_id = model_all_data.user_id.drop_duplicates().sample(frac=0.25)
train_data = model_all_data[~model_all_data.user_id.isin(test_user_id)]
test_data = model_all_data[model_all_data.user_id.isin(test_user_id)]
print('amount of total data: ', model_all_data.shape)
print('amount of training data: ', train_data.shape)
print('amount of testing data: ', test_data.shape)

# select model
train_x = train_data.drop(['user_id', 'product_id', 'label'], axis=1)
train_y = train_data['label']

classifiers = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier()]


def build_pipeline(classifier):
    steps = list()
    steps.append(('fill_nan', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.0)))
    steps.append(('down_sampling', RandomUnderSampler()))
    steps.append(('scalar', MinMaxScaler()))
    steps.append(('model', classifier))
    pipeline_ = Pipeline(steps=steps)
    return pipeline_


# print(build_pipeline(GradientBoostingClassifier()).get_params().keys())

for classifier in classifiers:
    pipeline = build_pipeline(classifier)
    scores = cross_val_score(pipeline, train_x, train_y, cv=5, scoring='f1')
    print(classifier.__class__.__name__, 'F1 score is %.3f (%.3f)' % (np.mean(scores)*100, np.std(scores)*100))
    print('_____________________')

print('From above results, we find the GradientBoosting have the highest F1 score,'
      'so we will use it to select our feature')
# select feature
pipeline2 = build_pipeline(GradientBoostingClassifier())
pipeline2.fit(train_x, train_y)

importance = pipeline2.steps[3][1].feature_importances_
features = train_x.columns.tolist()

importance1 = pd.DataFrame({'feature': features, 'importance': importance}).sort_values('importance', ascending=False)
top_15_feature = importance1.head(15)
print('the top 15 features are: ', top_15_feature)

train_x_feature = train_x[top_15_feature['feature']]


# train nerual network model
def nn_model():
    model = keras.Sequential([
        keras.layers.Dense(30, input_dim=15, activation='relu'),
        keras.layers.Dense(15, activation='relu'),
        keras.layers.Dense(5, activation='relu'),
        keras.layers.Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


NN_model = KerasClassifier(build_fn=nn_model, epochs=64, batch_size=32, verbose=0)
keras_model = nn_model()
print(keras_model.summary())

# model selection with selected top 15 features
classifiers.append(NN_model)

for classifier in classifiers:
    pipeline = build_pipeline(classifier)
    scores = cross_val_score(pipeline, train_x_feature, train_y, cv=5, scoring='f1')
    print(classifier.__class__.__name__, ': F1 value is %.3f (%.3f)' % (np.mean(scores)*100, np.std(scores)*100))
    print('_________________________')

# hyper-parameter tuning for GradientBoostingClassifier

parameters = {
    'model__n_estimators': [100, 150, 200],
    'model__max_depth': [4, 6, 8, 10],
    'model__min_samples_split': [6, 8, 10],
    'model__learning_rate': [0.005, 0.01, 0.02],
}

grid = GridSearchCV(build_pipeline(GradientBoostingClassifier()), cv=5, param_grid=parameters, scoring='f1')
grid.fit(train_x_feature, train_y)
print('best f1 value is %.3f' % grid.best_score_)
print('parameters are %s' % grid.best_params_)

boosting_model = build_pipeline(GradientBoostingClassifier(n_estimators=150, min_samples_split=6,
                                                           max_depth=4, learning_rate=0.005))
boosting_model.fit(train_x_feature, train_y)

# evaluate model on test data

test_x_feature = test_data.drop(['user_id', 'product_id', 'label'], axis=1)[top_15_feature['feature']]
test_y = test_data['label']
predict_y = boosting_model.predict(test_x_feature)
df_output = pd.DataFrame({'user_id': test_data.user_id, 'product_id': test_data.product_id,
                          'predict': predict_y, 'label': test_data.label})
print(df_output.head)
predict_y_prob = boosting_model.predict_proba(test_x_feature)[:, 1]

plot_confusion_matrix(boosting_model, test_x_feature, test_y, display_labels=['not reorder', 'reorder'], cmap=plt.cm.Blues)
plt.show()

print('model evaluation:')
acc = accuracy_score(test_y, predict_y)
print('accuracy: ', acc)
f1 = f1_score(test_y, predict_y)
print('F1 score: ', f1)
pre = precision_score(test_y, predict_y)
print('precision: ', pre)
rec = recall_score(test_y, predict_y)
print('recall: ', rec)
roc = roc_auc_score(test_y, predict_y)
print('ROC AUC: ', roc)

