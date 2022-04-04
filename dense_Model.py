from keras.models import Sequential
from keras.layers import Dense, Flatten
from pandas import read_csv
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.optimizers import SGD

dataframe = read_csv("antimalware.csv", skiprows=0)
dataset = dataframe.values

x = dataset[:, 0:328]
y = dataset[:, 328]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, shuffle='true', stratify=y)
xtrain = np.expand_dims(xtrain, axis=-1)
xtest = np.expand_dims(xtest, axis=-1)

from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10)
cvscores = []
cvscores2 = []
for train, test in kfold.split(x, y):
    model = Sequential()
    model.add(Dense(64, activation="relu", use_bias='True', input_shape=(328, 1)))
    model.add(Dense(16, activation="relu"))
    model.add(Flatten())
    model.add(Dense(300, activation="relu"))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(xtrain, ytrain,
                        batch_size=8,
                        epochs=100,
                        verbose=0)
    train_acc = model.evaluate(xtrain, ytrain, verbose=0)
    print("Loss:", train_acc[0], " Train accuracy:", train_acc[1])
    print()
    cvscores2.append(train_acc[1] * 100)

    testacc = model.evaluate(xtest, ytest)
    print("Loss:", testacc[0], " Test accuracy:", testacc[1])
    print()
    cvscores.append(testacc[1] * 100)

print("Train accuracy is: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores2), np.std(cvscores2)))
print("Test accuracy is: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

