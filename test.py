from sklearn import datasets, utils, ensemble
from sklearn.model_selection import train_test_split
import pandas as pd
import pprint

from ada_boost import boost

def main():
    # Call the preprocesser to get the data
    X_train, X_test, y_train, y_test = preprocess()

    # Call the nomral random forest (ensemble method)
    acc = trainForest(X_train, y_train, X_test, y_test)
    print('The accuracy of the random forest method for our wine data is: ', acc)

    # Now we are goint to call our implementation of the ada Boost
    boost(X_train, y_train, X_test, y_test, 11)

'''
    This method will Build a random forest classifier form the random forest library   
'''
def trainForest(training_data, training_target, test_data, test_target):
    # Simple random forest for comparison
    random_forest_classifier = ensemble.RandomForestClassifier(n_estimators=10, max_depth=1, random_state = 42)
    random_forest_classifier.fit(training_data, training_target.values.ravel())

    # Print metrics
    #print(random_forest_classifier.predict(test_data))
    #print(test_target.values.ravel())
    accuracy = random_forest_classifier.score(test_data, test_target)
    return accuracy

'''
    This method will load the data and pre-process it to convert it into a pandas df 
'''
def preprocess():
# We are going to use the wine dataset from sklearn to illustrate how boosting works
    wine_data = datasets.load_wine()

    # The wine_data is loaded into a dictionary format
    print('The wine data is loaded into a dict structure with keys:')
    for key in wine_data:
        print(key)

    numRows = wine_data.data.shape[0]
    numFeatures = wine_data.data.shape[1]
    print('The wine data has:', numRows, 'rows and ', numFeatures, 'features.')

    featureNames = wine_data.feature_names
    print('The wine data feature names are: ')
    pprint.pprint(featureNames)

    targetNames = wine_data.target_names
    print('The wine data targets are 3 types of wine: ')
    pprint.pprint(targetNames)

    # Metadata and information about the dataset can be obtainned via the .DESCR of the wine_data dict
    #print(wine_data.DESCR)

    # Construct a pandas dataframe which will contain the loaded data
    X = pd.DataFrame(wine_data.data, columns= featureNames)
    Y = pd.DataFrame(wine_data.target, columns=['Target'])

    # Train test split for our data 80% training 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    main()
