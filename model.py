from sklearn import svm
import pickle
# from util import plot_roc

class irisModel(object):
    
    def __init__(self):
        """Iris classifier
        Attributes:
            clf: sklearn classifier model
        """
        self.clf = svm.SVC(kernel='linear', C=1, probability=True)

    def train(self, X, y):
        """Trains the classifier to associate the label with the sparse matrix
        """
        # X_train, X_test, y_train, y_test = train_test_split(X, y)
        self.clf.fit(X, y)

    def predict_proba(self, X):
        """Returns probability for the binary class '1' in a numpy array
        """
        y_proba = self.clf.predict_proba(X)
        return y_proba[:, 1]

    def predict(self, X):
        """Returns the predicted class in an array
        """
        y_pred = self.clf.predict(X)
        return y_pred

    def pickle_clf(self, path='models/SentimentClassifier.pkl'):
        """Saves the trained classifier for future use.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
            print("Pickled classifier at {}".format(path))

    # def plot_roc(self, X, y, size_x, size_y):
    #     """Plot the ROC curve for X_test and y_test.
    #     """
    #     plot_roc(self.clf, X, y, size_x, size_y)