from numpy import count_nonzero, logical_and, logical_or, concatenate, mean, array_split, poly1d, polyfit, array
from numpy.random import permutation
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt


SVM_DEFAULT_DEGREE = 3
SVM_DEFAULT_GAMMA = 'auto'
SVM_DEFAULT_C = 1.0
ALPHA = 1.5


def prepare_data(data, labels, max_count=None, train_ratio=0.8):
    """
    :param data: a numpy array with the features dataset
    :param labels:  a numpy array with the labels
    :param max_count: max amout of samples to work on. can be used for testing
    :param train_ratio: ratio of samples used for train
    :return: train_data: a numpy array with the features dataset - for train
             train_labels: a numpy array with the labels - for train
             test_data: a numpy array with the features dataset - for test
             test_labels: a numpy array with the features dataset - for test
    """
    if max_count:
        data = data[:max_count]
        labels = labels[:max_count]

    train_data = array([])
    train_labels = array([])
    test_data = array([])
    test_labels = array([])

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    labeled_data = concatenate((data, labels.reshape(-1, 1)), axis=1)
    cutoff = round(labeled_data.shape[0] * train_ratio)
    shuffled = permutation(labeled_data)
    train_data = shuffled[0: cutoff]
    test_data = shuffled[cutoff + 1: shuffled.shape[0]]
    train_labels = train_data[:, -1]
    test_labels = test_data[:,-1]
    train_data = train_data[:,:-1]
    test_data = test_data[:,:-1]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return train_data, train_labels, test_data, test_labels


def get_stats(prediction, labels):
    """
    :param prediction: a numpy array with the prediction of the model
    :param labels: a numpy array with the target values (labels)
    :return: tpr: true positive rate
             fpr: false positive rate
             accuracy: accuracy of the model given the predictions
    """

    tpr = 0.0
    fpr = 0.0
    accuracy = 0.0


    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    for predict, label in zip(prediction, labels):
        if predict and label:
            tpr += 1
            accuracy += 1
        elif predict and not label:
            fpr += 1
        elif not predict and not label:
            accuracy += 1
            
    tpr /= labels[labels[:]==1].shape[0]
    fpr /= labels[labels[:]==0].shape[0]
    accuracy /= prediction.shape[0]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return tpr, fpr, accuracy


def get_k_fold_stats(folds_array, labels_array, clf):
    """
    :param folds_array: a k-folds arrays based on a dataset with M features and N samples
    :param labels_array: a k-folds labels array based on the same dataset
    :param clf: the configured SVC learner
    :return: mean(tpr), mean(fpr), mean(accuracy) - means across all folds
    """
    tpr = []
    fpr = []
    accuracy = []
    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    for i in range(len(folds_array)):
        train_data =  concatenate(folds_array[:i] + folds_array[i+1:])
        train_labels = concatenate(labels_array[:i] + labels_array[i+1:])
        test_data = folds_array[i]
        test_labels = labels_array[i]
        clf.fit(train_data, train_labels)
        prediction = clf.predict(test_data)
        i_tpr, i_fpr, i_acc = get_stats(prediction, test_labels)
        tpr.append(i_tpr)
        fpr.append(i_fpr)
        accuracy.append(i_acc)
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return mean(tpr), mean(fpr), mean(accuracy)


def compare_svms(data_array,
                 labels_array,
                 folds_count,
                 kernels_list=('poly', 'poly', 'poly', 'rbf', 'rbf', 'rbf',),
                 kernel_params=({'degree': 2}, {'degree': 3}, {'degree': 4}, {'gamma': 0.005}, {'gamma': 0.05}, {'gamma': 0.5},)):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :param kernels_list: a list of strings defining the SVM kernels
    :param kernel_params: a dictionary with kernel parameters - degree, gamma, c
    :return: svm_df: a dataframe containing the results as described below
    """
    svm_df = pd.DataFrame()
    svm_df['kernel'] = kernels_list
    svm_df['kernel_params'] = kernel_params
    svm_df['tpr'] = None
    svm_df['fpr'] = None
    svm_df['accuracy'] = None

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    data = array_split(data_array, folds_count)
    labels = array_split(labels_array, folds_count)
    tpr = []
    fpr = []
    accuracy = []
    for ktype, param in zip(kernels_list, kernel_params):
        if (ktype == 'poly'):
            SVM = SVC(kernel='poly', degree=param['degree'], gamma=SVM_DEFAULT_GAMMA)
        else:
            SVM = SVC(kernel='rbf', gamma=param['gamma'])
        i_tpr, i_fpr, i_acc = get_k_fold_stats(data, labels, SVM)
        tpr.append(i_tpr)
        fpr.append(i_fpr)
        accuracy.append(i_acc)
    
    svm_df['tpr'] = tpr
    svm_df['fpr'] = fpr
    svm_df['accuracy'] = accuracy

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return svm_df


def get_most_accurate_kernel():
    """
    :return: integer representing the row number of the most accurate kernel
    """
    best_kernel = 5
    return best_kernel


def get_kernel_with_highest_score():
    """
    :return: integer representing the row number of the kernel with the highest score
    """
    best_kernel = 5
    return best_kernel


def plot_roc_curve_with_score(df, alpha_slope=1.5):
    """
    :param df: a dataframe containing the results of compare_svms
    :param alpha_slope: alpha parameter for plotting the linear score line
    :return:
    """
    x = df.fpr.tolist()
    y = df.tpr.tolist()
    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    kernel = df.iloc[get_kernel_with_highest_score()]
    tpr = kernel['tpr']
    fpr = kernel['fpr']
    b = tpr - (1.5 * fpr)
    line_x = array([0.01 * i for i in range(100)])
    line_y = alpha_slope * line_x + b
    plt.plot(line_x, line_y)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim(0.95, 1.001)
    plt.scatter(x, y)
    plt.show()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def evaluate_c_param(data_array, labels_array, folds_count):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :return: res: a dataframe containing the results for the different c values. columns similar to `compare_svms`
    """

    res = pd.DataFrame()
    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    J = [1.0 /3, 2.0 / 3, 1.0]
    I = [10**i for i in range(1, -5, -1)]
    c_values = [ i * j for i in I for j in J]
    res['c_values'] = None
    res['tpr'] = None
    res['fpr'] = None
    res['accuracy'] = None   
    data = array_split(data_array, folds_count)
    labels = array_split(labels_array, folds_count)
    tpr = []
    fpr = []
    accuracy = []
    for c in c_values:
        SVM = SVC(C=c, kernel='rbf', gamma=0.5)
        i_tpr, i_fpr, i_acc = get_k_fold_stats(data, labels, SVM)
        tpr.append(i_tpr)
        fpr.append(i_fpr)
        accuracy.append(i_acc)
        
    res['c_values'] = c_values
    res['tpr'] = tpr
    res['fpr'] = fpr
    res['accuracy'] = accuracy
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return res


def get_test_set_performance(train_data, train_labels, test_data, test_labels):
    """
    :param train_data: a numpy array with the features dataset - train
    :param train_labels: a numpy array with the labels - train

    :param test_data: a numpy array with the features dataset - test
    :param test_labels: a numpy array with the labels - test
    :return: kernel_type: the chosen kernel type (either 'poly' or 'rbf')
             kernel_params: a dictionary with the chosen kernel's parameters - c value, gamma or degree
             clf: the SVM leaner that was built from the parameters
             tpr: tpr on the test dataset
             fpr: fpr on the test dataset
             accuracy: accuracy of the model on the test dataset
    """

    kernel_type = ''
    kernel_params = None
    clf = SVC(class_weight='balanced')  # TODO: set the right kernel
    tpr = 0.0
    fpr = 0.0
    accuracy = 0.0

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    kernel_type = 'rbf'
    kernel_params = {'gamma': 0.5}
    clf = SVC(class_weight='balanced', C=6.666667, kernel='rbf', gamma=0.5)
    clf.fit(train_data, train_labels)
    prediction = clf.predict(test_data)
    tpr, fpr, accuracy = get_stats(prediction, test_labels)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return kernel_type, kernel_params, clf, tpr, fpr, accuracy
