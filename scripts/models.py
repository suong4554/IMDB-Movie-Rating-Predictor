

def nbMultiNB(train_x, train_y, test_x):
  from sklearn.naive_bayes import MultinomialNB
  clf = MultinomialNB().fit(train_x, train_y)
  pred_y = clf.predict(test_x)
  return pred_y





def knn(train_x, train_y, test_x, size_k):
    from sklearn.neighbors import KNeighborsClassifier
    # apply k-Nearest-Neighbors Algorithm:
    knn = KNeighborsClassifier(n_neighbors=size_k)
    knn.fit(train_x, train_y)

    # predict the test results:
    pred_y = knn.predict(test_x)
    return pred_y


def kernel(train_x, train_y, test_x):
    from sklearn.kernel_ridge import KernelRidge
    # apply Linear Regression:
    mlp = KernelRidge()
    mlp.fit(train_x, train_y)

    # predict the results:
    y_prediction = mlp.predict(test_x)

    # return predictions:
    return y_prediction


def logreg(train_x, train_y, test_x):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(
        penalty='l2',
        multi_class="multinomial",
        solver='lbfgs',
        verbose=0,
        max_iter=1000,
        warm_start=False)
    lr.fit(train_x, train_y)
    y_prediction = lr.predict(test_x)
    return y_prediction



def sgd(train_x, train_y, test_x):
    from sklearn.linear_model import SGDClassifier
    # apply Linear Regression:
    model = SGDClassifier()
    model.fit(train_x, train_y)
    y_prediction = model.predict(test_x)
    return y_prediction

def passiveAgressive(train_x, train_y, test_x):
    from sklearn.linear_model import PassiveAggressiveClassifier
    # apply Linear Regression:
    model = PassiveAggressiveClassifier()
    model.fit(train_x, train_y)
    y_prediction = model.predict(test_x)
    return y_prediction


def perceptron(train_x, train_y, test_x):
    from sklearn.linear_model import Perceptron
    # apply Linear Regression:
    model = Perceptron()
    model.fit(train_x, train_y)
    y_prediction = model.predict(test_x)
    return y_prediction
