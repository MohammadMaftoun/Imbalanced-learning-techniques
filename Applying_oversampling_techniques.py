import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import (RandomOverSampler, SMOTE, SMOTENC, SMOTEN, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE)

data = pd.read_csv("/content/heart.csv")

data.head()

data.isna().sum()

data['output'].value_counts()

X = data.drop(['output'],axis=1)

y = data['output']

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

for train_ix, test_ix in kfold.split(X, y):
	x_train, x_test = X.iloc[train_ix], X.iloc[test_ix]
	y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

scaler = RobustScaler()

X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

oversamplers = {
    'RandomOverSampler': RandomOverSampler(random_state=42),
    'SMOTE': SMOTE(random_state=42),
    'SMOTENC': SMOTENC(random_state=42, categorical_features=[0]),
    'SMOTEN': SMOTEN(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42),
    'KMeansSMOTE': KMeansSMOTE(random_state=42),
    'SVMSMOTE': SVMSMOTE(random_state=42)
}

def evaluate_model(X_train_res, y_train_res):
    model = SVC(random_state=42)
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, output_dict=True)

results = {}
for name, sampler in oversamplers.items():
    try:
        print(f"Evaluating {name}...")
        if name == 'SMOTENC':
            X_res, y_res = sampler.fit_resample(X_train, y_train)
        else:
            X_res, y_res = sampler.fit_resample(X_train, y_train)
        results[name] = evaluate_model(X_res, y_res)
    except ValueError as e:
        print(f"Error with {name}: {e}")

for name, metrics in results.items():
    print(f"\n{name}:\n")
    print(metrics)

results_df = pd.DataFrame(results).T
print(results_df)
