import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# loads the data set
data = pd.read_csv("antimalware_test.csv")
print(data)

mlp1 = MLPClassifier(max_iter=6000)
parameter_space = {'hidden_layer_sizes': [(64, 16), (64, 32), (50, 50, 50), (100,), (100, 100, 50), (200,)],
                   'activation': ['tanh', 'relu', 'logistic', 'identity'], 'solver': ['sgd', 'adam'],
                   'alpha': [0.0001, 0.05, 0.01, 0.003, 0.08, 0.0005],
                   'learning_rate': ['constant', 'adaptive']}
gsclf = GridSearchCV(mlp1, parameter_space, n_jobs=-1, cv=5)

X = data.values[:, 0:204]
Y = data.values[:, 204]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, stratify=Y)
gsclf.fit(X_train, Y_train)

print(gsclf.best_params_)
