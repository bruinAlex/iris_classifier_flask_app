from model import irisModel
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split


def build_model():
    model = irisModel()

    dat = datasets.load_iris()

    data = pd.DataFrame(dat['data'], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    target = pd.Series(dat['target'])

    # Remap target ints to more useful text
    target_remap = {0: 'setosa',
                    1: 'versicolour',
                    2: 'virginica',
                   }
    target = target.replace(target_remap)

    del(dat)



    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.4, random_state=0)

    model.train(X_train, y_train)
    print('Model training complete')

    model.pickle_clf()

    # model.plot_roc(X_test, y_test)


if __name__ == "__main__":
    build_model()
