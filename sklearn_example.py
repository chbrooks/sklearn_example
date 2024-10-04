import random

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
import random
from sklearn.metrics import accuracy_score


def ZeroR(data) :
    c = Counter(data)
    return c.most_common(1)[0][0]

## return a random element from the dataset
## you implement this.
def RandR(data) :
    return random.choice(data)

## load in the iris dataset.
features, classifications = load_iris(return_X_y=True)

pairs = zip(features, classifications)
## use ZeroR and RandR to predict classifications.
ZeroRScore = 0
RandRScore = 0

for item in pairs :
    pred1 = ZeroR(classifications)
    pred2 = RandR(classifications)
    if pred1 == item[1] :
        ZeroRScore += 1
    if pred2 == item[1] :
        RandRScore += 1

### Let's split the training and test set.

X_train, X_test, y_train, y_test =  train_test_split(features, classifications,
                                                     test_size=0.2)
# how would we do this by hand?

# now let's use a real classifier.
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
results = gnb.predict(X_test)
print(list(zip(results,y_test)))



## five-fold cross-validation.
## If we use a built-in sklearn estimator, we can get this for free:


print(cross_val_score(gnb, features, classifications, cv=5))

# pipelining. Often we want to transform the inputs and then feed them
# into a classifier (or predictor, in sklearn lingo)

# for example, suppose we want to scale our data to have mean 0 and st.dev 1.
# we can do:

X_train_scaled = StandardScaler().fit(X_train).transform(X_train)
X_test_scaled = StandardScaler().fit(X_test).transform(X_test)

print(X_train_scaled)

# that's a pain to do by itself. Instead, we can create a pipeline.

pipe = make_pipeline(
    StandardScaler(),
    GaussianNB()
)
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe.fit(X_train, y_train)
print("pipelining")
print(accuracy_score(pipe.predict(X_test), y_test))