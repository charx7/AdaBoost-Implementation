from sklearn import datasets, utils, ensemble
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import preprocessing

import matplotlib 
import matplotlib.pyplot as plt

import pandas as pd
import pprint
from ada_boost import boost

def main():
    # Call the preprocesser to get the data
    X_train, X_test, y_train, y_test = preprocess()
    # PCA of the train data
    doPCA(X_train, y_train)

    # Call the normal random forest (ensemble method)
    acc = trainForest(X_train, y_train, X_test, y_test)
    print('The accuracy of the random forest method for our wine data is: ', acc)

    # Now we are goint to call our implementation of the ada Boost
    max_iters = 27
    showInfo = False
    for i in range(max_iters):
        print('Boosting for {0} Step(s)'.format(i+1))
        if i+1 == max_iters:
            showInfo = True 
        boost(X_train, y_train, X_test, y_test, i+1, showInfo)

def doPCA(X_train, y_train):
    # Standard Scaler
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(X_train)
    X_train_scaled = minmax_scale.transform(X_train)

    # Call the PCA object
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_train_scaled)
    X_reduced_df = pd.DataFrame(X_reduced, columns=['PC1', 'PC2'])
    y_train_reset_index = y_train.reset_index()
    
    pca_df = pd.concat([X_reduced_df, y_train_reset_index], axis=1, ignore_index=True)
    pca_df.columns = ['PC1', 'PC2', 'nein', 'label']
    zeroth_index = pca_df['label'] == 2
    print('The explained variance ratio is: ', pca.explained_variance_ratio_)
    print('The singular values(eigenvalues) of the pca matrix are: ', pca.singular_values_)

    # Plot the PCA
    plt.figure(figsize=(10,8))
    for label,marker,color in zip(
        range(0,3),('x', 'o', '^'),('blue', 'red', 'green')):
        current_index = pca_df['label'] == label
        temp_df = pca_df[current_index]
        plt.scatter(x = temp_df['PC1'],
                    y = temp_df['PC2'],
                    marker = marker,
                    color = color,
                    alpha = 0.6,
                    label='Class {}'.format(label)
                    )
    # Labeling
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title('Embeded 2 dimensional PCA representation of wine training data')
    plt.show()

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
    # Info display for the wine data
    print('Wine data info: \n', wine_data.DESCR)

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
