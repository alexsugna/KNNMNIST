"""
This file contains functions useful to building a machine learning model
to analyze the MNIST dataset.

Alex Angus, Zach Nussbaum
"""

import matplotlib.pyplot as plt 
import numpy as np
import math
from sklearn import metrics, datasets
from argparse import ArgumentParser
from PIL import Image

plt.rcParams.update({'font.size': 10}) #increase font size

def get_MNIST(resolution, trim=False, trim_fraction=None):
    """
    The function get_MNIST() returns the MNIST dataset. Each digit is a
    2d array of values, where each value represents the intensity of the pixel
    at that index.
    
    MNIST dictionary has keys:
        
        data : a 1-d array of pixel values
        
        target : the target values of each digit
        
        target_names : a list of possible targets (0-9)
        
        images : a 2-d version of data
        
        DESCR : additional information and references
    """
    if resolution is '8x8':
        digits = datasets.load_digits()
        del digits['DESCR']
        del digits['target_names']
        
    elif resolution is '28x28':
        digits = datasets.fetch_openml('mnist_784')
        del digits['feature_names']
        del digits['DESCR']
        del digits['details']
        del digits['categories']
        del digits['url']
    if trim:
        digits = trim28x28(digits, trim_fraction)
    return digits


def reduce_features(digits, new_resolution):
    
    """
    Reduces the resolution, and therefore the number of features, of high 
    resolution images. Currently only works from 28x28 resolution
    
    params:
        digits: ndarray of images
        new_resolution: string specifying the desired new reolution
        
    returns:
        reduced_digits: ndarray of reduced resolution images
    """
    if len(new_resolution) < 5:
        new_res = int(new_resolution[0:1])
    else:
        new_res = int(new_resolution[0:2]) #get new resolution from string input
        
    old_res = int(np.sqrt(np.shape(digits[0])[0])) #old resolution
    #reduced_digits = np.zeros(len(digits)) #new reduced array
    reduced_digits = []
    for digits_index in range(len(digits)):
        #Image only works with 2d arrays, so we have to reshape to create an Image object
        digit = Image.fromarray(np.reshape(digits[digits_index], (old_res, old_res)))
        #use resize function and flatten back into 1d array.
        #reduced_digits[digits_index] = list(np.asarray(digit.resize((new_res, new_res))).flatten())
        reduced_digits.append(np.asarray(digit.resize((new_res, new_res))).flatten())
    return np.array(reduced_digits)
        

def trim28x28(digits, fraction):
    """
    reduces the size of the dataset digits by the fraction 
    specified.
    """
    features = np.array(digits.data)
    targets = np.array(digits.target, dtype=np.int)
    #shuffle features and targets with same seed
    order = np.random.permutation(len(features))
    
    digits.update({'features' : features[order]})
    digits.update({'targets' : targets[order]})

    length = int(len(digits.target) * fraction)
    
    trimmed_data = digits.data[:length]
    trimmed_targets = digits.target[:length]
    
    digits.update({'data' : trimmed_data})
    digits.update({'target' : trimmed_targets})
    
    return digits


def get_target_data(digits_dictionary):
    """
    returns a target array and a data array
    """
    return digits_dictionary['target'], digits_dictionary['data']


def train_test_split(digits, test_size=.15):
    """
    shuffles and creates a train and test set
        array is in column order of images, data, target
    
    INPUTS:
        digits: is the dictionary of MNIST data
        test_size: size of the test size
        
    OUTPUTS:
        x_test, y_test, x_train, y_train
    """
    features = np.array(digits.data)
    targets = np.array(digits.target, dtype=np.int)

    order = np.random.permutation(len(features))
    
    x, y = features[order], targets[order]

    test_size = int(x.shape[0] * 2 * test_size)

    x_test, y_test, x_train, y_train = x[:test_size], y[:test_size], x[test_size:], y[test_size:]
    
    return x_test, y_test, x_train, y_train, order
    

def show_digit(digit):
    """
    the function show_digit() displays a 2-d array of pixel values, where digit
    is the 2-d array

    params:
        digit: a 2-d array image array
    """
    
    if len(np.shape(digit)) < 2:
        axis_length = int(np.sqrt(len(digit)))
        reshaped_digit = np.reshape(digit, (axis_length, axis_length))
        plt.imshow(reshaped_digit, cmap='Greys')
        plt.show(block=False)
    else:
        plt.imshow(digit, cmap='Greys')
        plt.show(block=False)

    plt.pause(1)
    plt.close()
    
    
def show_digits(digits, images_per_row=10):
    """
    Same as show_digit except displays several images at once.
    """

    size = int(np.sqrt(np.shape(digits)[1]))
    images_per_row = min(len(digits), images_per_row)
    images = [digit.reshape(size,size) for digit in digits]
    try:
        n_rows = (len(digits) - 1) // images_per_row + 1
    except ZeroDivisionError: #if there are no images that meet the requirement, plot zeros
        plt.imshow(np.zeros((size, size)), cmap='Greys')
        plt.axis('off')
        return
    row_images = []
    n_empty = n_rows * images_per_row - len(digits) 
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap='Greys')
    plt.axis('off')
    
    
class Scaler:
    def __init__(self):
        self.params = {}

    def fit_transform(self, data):
        """
        Fits and transforms the data and stores 
            a copy of params used to transform the data
        
        x_normalized = (x - mean(x)) / std(x)

        params:
            data: a numpy array of all the data

        returns:
            copy: a normalized data set
        """
        copy = np.copy(data)

        mean = np.mean(copy)
        std = np.std(copy)

        copy = copy - mean

        copy = np.divide(copy, std, where=std!=0)

        self.params.update({'1d': (mean, std)})

        return copy

    def transform(self, data):
        """
        Transforms a dataset using the fitted params

        params:
            data: a numpy array

        retuns:
            full_data: a normalized array
        """
        (mean, std) = self.params.get('1d')

        copy = np.copy(data)

        copy = (copy - mean)
        copy = np.divide(copy, std, where=std!=0)

        return copy


    def inverse_transform(self, data):
        """
        Performs an inverse transform on the data using
            the params stored

        params:
            data: a normalized numpy array
        
        returns:
            full_data: a denormalized numpy array
        """
        copy = np.copy(data)

        (mean, std) = self.params.get('1d')

        copy = np.multipy(copy, std, where=std!=0)
        copy += mean

        return copy


class KNN(object):
    """
    An implementation of the K-Nearest Neighbors algorithm

    methods:
        train(X, y): train the KNN classifier
        distance(X_test): calculates the Euclidean distance between
            two arrays
        predict(X_test, k): predict the labels of X_test using k neighbors

    attributes:
        X: a numpy array of training features
        y: a numpy array of training targets
    """
    def __init__(self):
        pass

    def train(self, X, y):
        """
        'training' for KNN. Just store the training data

        params:
            X: a numpy array of features
            y: a numpy array of targets
        """
        self.X = X
        self.y = y
        

    def distance(self, X_test):
        """
        Calculates the Euclidean distance between two arrays

        params:
            X_test: a numpy array of test examples to predict on

        returns:
            dists: a numpy array of distances where i,j refers to the distance
                between test example i and train example j
        """
        num_train = self.X.shape[0]
        num_test = X_test.shape[0]

        dists = np.zeros((num_test, num_train))

        x2 = np.sum(X_test**2,axis=1)
        y2 = np.sum(self.X**2,axis=1)
        xy = np.dot(X_test, self.X.T)
   
        dists = np.sqrt(np.abs(x2[:,np.newaxis] + y2 - xy*2))
        return dists
    

    def predict(self, X_test, k):
        """
        Predicts the KNN for all test examples

        params:
            X_test: a numpy array of test examples
            k: int of nearest neighbors to find

        returns:
            y_pred: a numpy array of predictions
                of size (D,1), where D is the size of X_test
        """
        distances = self.distance(X_test)

        num_test = X_test.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            
            current_image = distances[i, :]

            sorted_row = np.argsort(current_image)

            closest_y = self.y[sorted_row[:k]]
            
            y_pred[i] = np.argmax(np.bincount(closest_y.astype(int)))

        return y_pred
    
    
class KNN_manhattan(KNN):
    def distance(self, X_test):
        """
        KNN classifier that uses the manhattan distance rather than the Euclidean 
        distance.
        
        Manhattan distance is defined as:
            
            M_distance(p, q) = sum(abs(p_i - q_i))
            
        params:
            X_test: a numpy array of test examples
            
        returns:
            the manhattan distance between the test set and the training set 
            instances. 
        
        """
        return np.abs(X_test[:,np.newaxis] - self.X).sum(-1)

class WKNN(KNN):
    def predict(self, X_test, k):
        """
        Predicts the WKNN for all test examples, where WKNN is the weighted 
        version of KNN.

        params:
            X_test: a numpy array of test examples
            k: int of nearest neighbors to find

        returns:
            y_pred: a numpy array of predictions
                of size (D,1), where D is the size of X_test
        """
        distances = self.distance(X_test)

        num_test = X_test.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            
            current_image = distances[i, :]

            sorted_distances = np.argsort(current_image)[:k]

            closest_y = self.y[sorted_distances[:k]]

            weights = np.zeros((1, 10))

            for j in range(closest_y.shape[0]):
                # make sure we're not dividing by zero, or close to 0
                if math.isclose(sorted_distances[j], 0):
                    sorted_distances[j] = 1
                weights[:, closest_y[j].astype(int)] += 1/sorted_distances[j]

            y_pred[i] = np.argmax(weights)

        return y_pred
    

class WKNN_manhattan(WKNN, KNN_manhattan):
    """
    Combines the weighted KNN model with the manhattan distance KNN model.
    """
    def distance(self, X_test):
        return KNN_manhattan.distance(self, X_test)

    def predict(self, X_test, k):
        return WKNN.predict(self, X_test, k)


def kfold_validation(x_train, y_train, num_folds=5, classifier=KNN):
    """
    Performs KFold Validation on the training set and returns 

    params:
        x_train: a numpy array of training features
        y_train: a numpy array of training targets
        num_folds: the number of folds to cross validate on

    returns:
        k_accuracies: a dictionary of number of k to list of accuracies for 
            that value of k
    """
    k_choices = [i for i in range(1, 10)]

    X_train_folds = np.array_split(x_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)

    k_accuracies = {}

    for i in range(len(k_choices)):
        k = k_choices[i]
        accuracies = []
        for j in range(num_folds):
            current_x_train = np.concatenate(tuple(X_train_folds[k] for k in range(num_folds) if k != j))
            current_y_train = np.concatenate(tuple(y_train_folds[k] for k in range(num_folds) if k != j))

            current_x_val = X_train_folds[j]
            current_y_val = y_train_folds[j]

            knn = classifier()
            knn.train(current_x_train, current_y_train)

            preds = knn.predict(current_x_val, k)
            acc = (preds == current_y_val).mean()

            accuracies.append(acc)

        k_accuracies[k] = accuracies

    return k_accuracies


def confusion_matrix(y_pred, y_train, display=False, classifier="KNN", resolution='8x8'):
    """
    returns the confusion matrix for a given set of predictions using the 
    sklearn confusion_matrix function. The matrix will be displayed if 
    display = True
    
    also analyzes the errors that our model is making and displays an error 
    matrix
    """
    conf_mx = metrics.confusion_matrix(y_train, y_pred)
    
    if display:
        plt.matshow(conf_mx, cmap=plt.cm.gray)
        plt.xlabel("Labels")
        plt.title('{} Confusion Matrix for ({} pixels)'.format(classifier, resolution), y=1.08)
        plt.ylabel("Predictions")
        plt.savefig('figs/{} confusion matrix for ({} pixels)'.format(classifier, resolution))
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    #divide each class by the number of images we have to train on of that class
    sum_rows = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / sum_rows
    np.fill_diagonal(norm_conf_mx, 0) #look at only where we make errors
    
    figure = plt.figure()
    plt.tight_layout()
    axis = figure.add_subplot(111)
    axis.set_xlabel("Labels")
    axis.set_ylabel("Predictions")
    figure.suptitle("Error Matrix {}".format(classifier), y=1.08)
    error_axis = axis.matshow(norm_conf_mx)
    figure.colorbar(error_axis)
    plt.savefig('figs/{} error matrix for ({} pixels)'.format(classifier, resolution))
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    
    return conf_mx


def plot_bad_predictions(normalized_train, y_train, y_pred, classifier="KNN", resolution='8x8'):
    """
    Plots the bad predictions for a classifier and saves the figure

    params:
        normalized_train: the training data
        y_train: the training labels
        y_pred: the predicted labels
        classifier: the classifier that was used to predict
        resolution: the size of the images
    """

    
    a = int(input("Bad Prediction 1: "))
    b = int(input("Bad Prediction 2: "))
    
    y_train_ints = y_train.astype(int)
    y_pred_ints = y_pred.astype(int)
    
    X_aa = normalized_train[(y_train_ints == a) & (y_pred_ints == a)]
    X_ab = normalized_train[(y_train_ints == a) & (y_pred_ints == b)]
    X_ba = normalized_train[(y_train_ints == b) & (y_pred_ints == a)]
    X_bb = normalized_train[(y_train_ints == b) & (y_pred_ints == b)]

    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(221); show_digits(X_aa[:25], images_per_row=5)
    ax2 = fig.add_subplot(222); show_digits(X_ab[:25], images_per_row=5)
    ax3 = fig.add_subplot(223); show_digits(X_ba[:25], images_per_row=5)
    ax4 = fig.add_subplot(224); show_digits(X_bb[:25], images_per_row=5)
    
    ax1.title.set_text('Correctly Classified as ' + str(a))
    ax2.title.set_text('Incorrectly Classified as ' + str(b))   
    ax3.title.set_text('Incorrectly Classified as ' + str(a))
    ax4.title.set_text('Correctly Classified as ' + str(b))
    
    plt.savefig('figs/{} Bad Predictions for ({} pixels)'.format(classifier, resolution))
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def plot_kfold_validation(k_accuracies, resolution, classifier="KNN"):
    """
    Plots the kfold validation accuracy and error

    params:
        k_accuracies: dictionary of k mapped to a list of accuracies
        resolution: string denoting the resolution of the images
        classifier: a string of which classifier you used to validate on
    """
    for k in sorted(k_accuracies):
        accuracies = k_accuracies[k]
        for acc in accuracies:
            print("k: {} accuracy: {}".format(k, acc))

    k_choices = [i for i in range(1, 10)]
    for k in k_choices:
        accuracy = k_accuracies[k]
        plt.scatter([k] * len(accuracy), accuracy)

    accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k,v in sorted(k_accuracies.items())])
    plt.tight_layout()
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title("{} Accuracy vs Number of Neighbors ({} pixels)".format(classifier, resolution))
    plt.savefig("figs/{} Accuracy vs Number of Neighbors ({} pixels)".format(classifier, resolution))
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def compare_classifiers(resolution, reduced_resolution):
    """
    Cross validates and compares the KNN to the Weighted KNN

    params:
        resolution: string denoting the resolution of the fetched images
        reduced_resolution: string denoting the desired downsized resolution

    returns:
        x_test, y_test, x_train, y_train: numpy arrays of the split training and test
            features and labels
    """
    trim_fraction = 0.1
    if resolution == "28x28":
        trim = True
    else:
        trim = False

    digits = get_MNIST(resolution, trim=trim, trim_fraction=trim_fraction)
    
    large_x_test, y_test, large_x_train, y_train, order = train_test_split(digits)
    #reduce the resolution
    x_test = reduce_features(large_x_test, reduced_resolution)
    x_train = reduce_features(large_x_train, reduced_resolution)
    scaler = Scaler()
    normalized_train = scaler.fit_transform(x_train)
    normalized_test = scaler.transform(x_test)
    
    k_accuracies = kfold_validation(normalized_train, y_train)
    plot_kfold_validation(k_accuracies, reduced_resolution)

    k_accuracies = kfold_validation(normalized_train, y_train, classifier=WKNN)
    plot_kfold_validation(k_accuracies, reduced_resolution, classifier="WKNN")
    
    k_accuracies = kfold_validation(normalized_train, y_train, classifier=KNN_manhattan)
    plot_kfold_validation(k_accuracies, reduced_resolution, classifier="KNN_manhattan")
    
    k_accuracies = kfold_validation(normalized_train, y_train, classifier=WKNN_manhattan)
    plot_kfold_validation(k_accuracies, reduced_resolution, classifier="WKNN_manhattan")
    
    np.savetxt('data/x_train_{}.txt'.format(reduced_resolution), normalized_train)
    np.savetxt('data/y_train_{}.txt'.format(reduced_resolution), y_train)
    np.savetxt('data/x_test_{}.txt'.format(reduced_resolution), normalized_test)
    np.savetxt('data/y_test_{}.txt'.format(reduced_resolution), y_test)

    return x_test, y_test, x_train, y_train


def test_classifiers(resolution, reduced_resolution, new_train_test=False):
    """
    Tests the Weighted KNN and KNN classifier based on the resolution
    Either loads the saved training/test data from validation before
        or you can train and validate on a new model to then predict on 
        the test set by passing -n as in argument when you call the file

    params:
        resolution: string denoting the resolution of the images
        reduced_resolution: string denoting the desired downsized resolution
    """
    parser = ArgumentParser()

    parser.add_argument('-n', '--new', action='store_true')

    args = parser.parse_args()

    if args.new or new_train_test:
        x_test, y_test, x_train, y_train = compare_classifiers(resolution, reduced_resolution)
    else:
        x_train = np.loadtxt('data/x_train_{}.txt'.format(reduced_resolution))
        y_train = np.loadtxt('data/y_train_{}.txt'.format(reduced_resolution))
        x_test = np.loadtxt('data/x_test_{}.txt'.format(reduced_resolution))
        y_test = np.loadtxt('data/y_test_{}.txt'.format(reduced_resolution))

    while True:
        try:
            k_knn = int(input("Number of nearest for KNN: "))
        except:
            print("please input an integer")

        try:
            k_wknn = int(input("Number of nearest for WKNN: "))
            
        except:
            print("please input an integer")
            
        try:
            k_mknn = int(input("Number of nearest for KNN_manhattan: "))
            
        except:
            print("please input an integer")
        
        try:
            k_wmknn = int(input("Number of nearest for WKNN_manhattan: "))
            if k_knn or k_mknn or k_mknn:
                break
        except:
            print("please input an integer")
                
    
    knn = KNN()
    knn.train(x_train, y_train)
    knn_pred = knn.predict(x_test, k_knn)
    acc = (knn_pred == y_test).mean()
    print("KNN with K {} had accuracy {}".format(k_knn, acc))
    confusion_matrix(knn_pred, y_test, display=True, resolution=reduced_resolution)
    plot_bad_predictions(x_test, y_test, knn_pred, resolution=reduced_resolution)

    wknn = WKNN()
    wknn.train(x_train, y_train)
    wknn_pred = wknn.predict(x_test, k_wknn)
    acc = (wknn_pred == y_test).mean()
    print("WKNN with K {} had accuracy {}".format(k_wknn, acc))
    confusion_matrix(wknn_pred, y_test, display=True, classifier="WKNN", resolution=reduced_resolution)
    plot_bad_predictions(x_test, y_test, wknn_pred, classifier="WKNN", resolution=reduced_resolution)
    
    mknn = KNN_manhattan()
    mknn.train(x_train, y_train)
    mknn_pred = mknn.predict(x_test, k_wknn)
    acc = (mknn_pred == y_test).mean()
    print("KNN_manhattan with K {} had accuracy {}".format(k_mknn, acc))
    confusion_matrix(mknn_pred, y_test, display=True, classifier="KNN_manhattan", resolution=reduced_resolution)
    plot_bad_predictions(x_test, y_test, mknn_pred, classifier="KNN_manhattan", resolution=reduced_resolution)
    
    wmknn = WKNN_manhattan()
    wmknn.train(x_train, y_train)
    wmknn_pred = wmknn.predict(x_test, k_wmknn)
    acc = (wmknn_pred == y_test).mean()
    print("WKNN_manhattan with K {} had accuracy {}".format(k_wmknn, acc))
    confusion_matrix(wmknn_pred, y_test, display=True, classifier="WKNN_manhattan", resolution=reduced_resolution)
    plot_bad_predictions(x_test, y_test, wmknn_pred, classifier="WKNN_manhattan", resolution=reduced_resolution)
    
def main():
    """
    Main function that runs the testing of classifiers
    """
    resolution = "28x28"
    reduced_resolution = "16x16"
    test_classifiers(resolution, reduced_resolution, new_train_test=True)
    
if __name__ == "__main__":
    main()