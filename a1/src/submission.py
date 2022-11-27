#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar, DefaultDict
from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 1: binary classification
############################################################

############################################################
# Problem 1a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    #pass
    # ### START CODE HERE ###
    words = x.split(' ')
    vector = {}#{words[0]:1}
    for word in words:
        if word in vector:
            vector[word] = vector[word] + 1
        else:
            vector[word] = 1
    return vector
    # ### END CODE HERE ###


############################################################
# Problem 1b: stochastic gradient descent

T = TypeVar("T")


def learnPredictor(
    trainExamples: List[Tuple[T, int]],
    validationExamples: List[Tuple[T, int]],
    featureExtractor: Callable[[T], FeatureVector],
    numEpochs: int,
    eta: float,
) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    """
    weights = {}  # feature => weight
    # ### START CODE HERE ###
    trainError = 0
    testError = 0
    trainVectors = []
    validationVectors = []

    scores = []
    loss_h = []
    gradientloss_h = {}
    #split examples from predictions
    #do I need to have 1 base vocab set here 
    for example in trainExamples:
        trainVectors.append((featureExtractor(example[0]),example[1]))
    for val in validationExamples:
        validationVectors.append((featureExtractor(val[0]), val[1]))
    
    
    #initialize the weights
    for vec in trainVectors:
        for word in vec[0]:
            if word not in weights:
                weights[word] = 0
    
    for epoch in range(numEpochs):
        #for vec in trainVectors
             #sparse vectorize the example
        for vector, ans in trainVectors:
            score = dotProduct(weights,vector)
            #scores.append(score)

            loss = max((1-score*ans),0)
            #loss_h.append(loss) 
            if loss != 0:
                for word, pred in list(vector.items()):
                    if word in gradientloss_h:
                        gradientloss_h[word] = gradientloss_h[word] + pred*ans#.append(1/len(trainExamples) * sum(-score*ans if loss > 0 else 0))
                    else:
                        gradientloss_h[word] = pred*ans
            # still need to aggregate the sparse representations together for gradientloss_h 
        #this is wrong I need to understand how I get losses per word to 
        #apply to the weights w/ increment instead of /example as it is presently
        for word,pred in list(gradientloss_h.items()):
            gradientloss_h[word] = pred/len(trainVectors)
        
        #calc the loss function here
        increment(weights, eta, gradientloss_h)
        
        #This is just wrong cause I don't have a predictor which I'm guessing is supposed to be
        #
        #def predictor(x):
        #    pred = dotProduct(weights,x)
        #    return pred
        #trainError = evaluatePredictor(trainVectors, predictor)
        #testError = evaluatePredictor(validationVectors, predictor)
        #for ex in trainExamples:
            
        print("Epoch: %f Training Error: %f Test Error: %f", epoch, trainError, testError)
    # ### END CODE HERE ###
    return weights


############################################################
# Problem 1c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        phi = None
        y = None
        # ### START CODE HERE ###
        phi = {}
        y = 0
        eLen = random.randint(1,7)
        #create the example vector
        for len in range(eLen):
            word, weight = random.choice(list(weights.items()))
            if word in phi:
                phi[word] = phi[word] + 1
            else: 
                phi[word] = 1
        y = dotProduct(phi,weights)
        #for feature in range(len(WeightVector)):
        #    phi.append(random.random())
        #
        #y = dotProduct(phi,WeightVector) + random.random
        
        #I could make this ternary
        if(y >= 1): y=1
        else: y=-1
        # ### END CODE HERE ###
        return (phi, y)

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 1d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x):
        pass
        # ### START CODE HERE ###
        fvec = {}
        x = x.replace(' ','')
        for i in range(len(x)-n+1):
            str = x[i:i+n]
            if str in fvec:
                fvec[str] = fvec[str] + 1
            else:
                fvec[str] = 1
        #    for ftr in fvec:
        #        if str == ftr.key:
        #            ftr.value = ftr.value + 1
        #        if ftr.key == fvec[len(fvec)-1]:
        #            fvec.append(str,1)

        return fvec
        # ### END CODE HERE ###

    return extract


############################################################
# Problem 1e:
#
# Helper function to test 1e.
#
# To run this function, run the command from termial with `n` replaced
#
# $ python -c "from solution import *; testValuesOfN(n)"
#


def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be submitted.
    """
    trainExamples = readExamples("polarity.train")
    validationExamples = readExamples("polarity.dev")
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(
        trainExamples, validationExamples, featureExtractor, numEpochs=20, eta=0.01
    )
    outputWeights(weights, "weights")
    outputErrorAnalysis(
        validationExamples, featureExtractor, weights, "error-analysis"
    )  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    validationError = evaluatePredictor(
        validationExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    print(
        (
            "Official: train error = %s, validation error = %s"
            % (trainError, validationError)
        )
    )


############################################################
# Problem 2b: K-means
############################################################


def kmeans(
    examples: List[Dict[str, float]], K: int, maxEpochs: int
) -> Tuple[List, List, float]:
    """
    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    """
    # ### START CODE HERE ###
    centroids = []
    
    for k in range(K):
        #randomly initilize the centroids with existing points
        centroids.append(random.choice(examples))
    new_centroids = centroids

    #mu = assignments, phi, loss
    #centroid number for given example
    #Z has same length as examples
    best_distance = 3000
    i = 0
    for epoch in range(maxEpochs):
        z = []
        #assign z for each of the examples
        for example in examples:
            for ctr in range(len(centroids)):
                cur_distance = sum([(centroids[ctr][k]-example[k])**2 for k in centroids[ctr]])
                if cur_distance < best_distance:
                    i = ctr #i is the closest centroid to the current example
                    best_distance = cur_distance
            z.append(i)
            best_distance = 3000
        #make a blank set of centroids again
        for i in range(len(centroids)):
            for k,v in centroids[i].items():
                new_centroids[i][k] = 0
        #relocate the centroids
        for centroid in range(len(new_centroids)):
            i = 0 # i is the count of examples custered to the centroid
            for ex in range(len(z)):
                if z[ex] == centroid:
                    i = i+1
                    for k,v in new_centroids[centroid].items():
                        new_centroids[centroid][k] = v + examples[ex][k]
                    #new_centroids[centroid] = new_centroids[centroid] + examples[ex]
            for k,v in new_centroids[centroid].items():
                new_centroids[centroid][k] = v/i
        #exit early on convergence 
        centroids = new_centroids
        #if sum([new_centroids[k] - centroids[k] for k in centroids]) == 0:
        #    break
        #else: 
        #    centroids = new_centroids
    #return centroids, example assignments and reconstruction loss        
    return (centroids,z,4)
    # ### END CODE HERE ###
