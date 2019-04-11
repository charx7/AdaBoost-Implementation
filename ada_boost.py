import numpy as np
import random
import math
from sklearn import ensemble

class classifier():
    # Class constructor
    def __init__ (self, data, target, classifier, weights, testX, testY):
        self.clf         = classifier
        self.weights     = weights
        self.nextWeights = None
        self.testX       = testX
        self.testY       = testY 
        self.X           = data
        self.Y           = target
        self.sampleX     = None
        self.sampleY     = None
        self.predictions = None
        self.alpha_m     = None

    # Sample method for sampling with replacement
    def sampleData(self, data, target, seed,  doSample):
        numSamples = data.shape[0]
        if doSample == True:
            # Sample with replacement according to the current weights
            sampleTrain  = data.sample(n=numSamples, weights=self.weights, random_state=seed)
            sampleTarget = target.sample(n=numSamples, weights=self.weights, random_state=seed)
            self.sampleX = sampleTrain
            self.sampleY = sampleTarget
        else:
            self.sampleX = data
            self.sampleY = target

    # The classification method of the current classifier
    def classify(self):
        predictions = self.clf.predict(self.X.values)
        self.predictions = predictions

    # This method will compute the error of the classifier alpha_{m}
    def computeError(self):
        numSamples = self.X.shape[0]
        #print(self.predictions)
        #print(self.Y.values.ravel())
        errSum = 0
        for i in range(numSamples):
            currentPrediction = self.predictions[i]
            currentRealValues = self.Y.values.ravel()[i]
            currentWeight     = self.weights[i]
            if currentPrediction != currentRealValues:
                errSum = errSum + currentWeight
        
        # Calculate clf error
        alpha_m = math.log((1-errSum) / errSum)
        self.alpha_m = alpha_m
        #print('the sum is: ', errSum)
        print('The voting amount of alpha_m of the current clf is : ', alpha_m)
        
    # This method will change the weights of the classifier for the next iteration
    def set_next_weights(self):
        numSamples = self.X.shape[0]
        newWeights = []
        for i in range(numSamples):
            currentWeight = self.weights[i]
            currentPrediction = self.predictions[i]
            currentRealValues = self.Y.values.ravel()[i]
            classifierWeight  = self.alpha_m
            if currentPrediction != currentRealValues:
                # if the current value is different to the predicted value increase the weight of the example
                newWeight = currentWeight * math.exp(classifierWeight)
            else:
                # if the values are correct decrease the importance of the example
                newWeight = currentWeight * math.exp(-classifierWeight)
            # append to the new weights vector
            newWeights.append(newWeight)
            # numpyfy it
            numpyifiedweights = np.array(newWeights)
            # Normalize the vector
            normalized_new_weights = np.dot(numpyifiedweights, 1/np.sum(numpyifiedweights))
        # Sanity check for the sum
        #print(np.sum(normalized_new_weights))
        self.nextWeights = normalized_new_weights

    # The fit method for the current classifier
    def fit(self):
        # Fit the rf to the current sampled data
        self.clf.fit(self.sampleX, self.sampleY.values.ravel())
        # Classify the fitted data and save it into the obj
        self.classify()
        #print(self.sampleY.head(5))
        #print(self.sampleX.head(5))
        
def boost(training_data, training_target, test_data, test_target, boostingSteps):
    numTrainingSamples = training_data.shape[0]
    # Initialize all weights to be 1/N
    weights = np.ones(numTrainingSamples)
    weights = np.dot(weights, 1/numTrainingSamples)
    
    ensembles_vector = []
    for i in range(boostingSteps):
        # Note a stump is just a tree with depth 1 :D
        current_clf = classifier(
            training_data,
            training_target,
            ensemble.RandomForestClassifier(n_estimators=10,max_depth=1, random_state = i+1),
            weights,
            test_data,
            test_target
        )
        
        # if it is the first iteration then do not sample else sample with replacement
        current_seed = 42 + i
        if i == 0:
            current_clf.sampleData(training_data, training_target, current_seed, doSample = False)
        else:
            current_clf.sampleData(training_data, training_target, current_seed, doSample = True)

        # Fit the classifier to the data
        current_clf.fit()
        
        # Compute the error of the classifier
        current_clf.computeError()

        # Assign new weights for the next iteration
        current_clf.set_next_weights()
        weights = current_clf.nextWeights
        
        # Add the new classifiers to the classifiers vector
        ensembles_vector.append(current_clf)
    
    print(ensembles_vector)

    
def main():
    print('im main')

if __name__ == '__main__':
    main()