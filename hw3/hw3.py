import numpy as np
np.random.seed(42)

####################################################################################################
#                                            Part A
####################################################################################################

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.class_value = class_value
        data = dataset[dataset[:,-1]==class_value]
        self.mean = np.array([np.mean(data[:,c], dtype='float64') for c in range(data.shape[1] - 1)])
        self.std = np.array([np.std(data[:,c], dtype='float64') for c in range(data.shape[1] - 1)])
        self.prior = data.shape[0] / dataset.shape[0]
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        p = 1.0
        for c in range(x.shape[0] - 1):
#            p *= (1.0 / (np.sqrt(2 * np.pi * np.square(self.std[c])))) * np.exp(-0.5 * np.square((x[c] - self.mean[c])/self.std[c]))
            p *= normal_pdf(x[c], self.mean[c], self.std[c])
        return p
            
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.prior
    
class MultiNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.class_value = class_value
        data = dataset[dataset[:,-1]==class_value]
        self.mean = np.array([np.mean(data[:,c], dtype='float64') for c in range(data.shape[1] - 1)])
        self.cov = np.cov([dataset[:, i] for i in range(dataset.shape[1] - 1)])
        self.prior = data.shape[0] / dataset.shape[0]
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        return multi_normal_pdf(np.array([x[i] for i in range(x.shape[0] - 1)]), self.mean, self.cov)
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.prior
    
    

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    return (1.0 / (np.sqrt(2 * np.pi * np.square(std)))) * np.exp(-0.5 * np.square((x - mean)/std))

    
def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variante normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    return np.power(2 * np.pi, -0.5 * x.shape[0]) * np.power(np.linalg.det(cov), -0.5) * np.exp((-0.5 * np.dot((x-mean).T, (np.dot(np.linalg.inv(cov), (x-mean))))))


####################################################################################################
#                                            Part B
####################################################################################################
EPSILLON = 1e-6 # == 0.000001 It could happen that a certain value will only occur in the test set.
                # In case such a thing occur the probability for that value will EPSILLON.
    
class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with la place smoothing.
        
        Input
        - dataset: The dataset from which to compute the probabilites (Numpy Array).
        - class_value : Compute the relevant parameters only for instances from the given class.
        """
        
        self.class_value = class_value
        self.data = dataset[dataset[:,-1]==class_value]
        self.possible_values_per_attr = np.array([np.unique(dataset[:,i]).shape[0] for i in range(dataset.shape[1] - 1)])
        self.prior = self.data.shape[0] / dataset.shape[0]
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        p = 1.0
        for i in range(x.shape[0] - 1):
            nij = self.data[self.data[:,i]==x[i]]
            if (not nij.any()):
                nij = EPSILLON
            else:
                nij = nij.shape[0]
            p *= (nij + 1 / (self.prior + self.possible_values_per_attr[i]))
        return p
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.prior

    
####################################################################################################
#                                            General
####################################################################################################            
class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a postreiori classifier. 
        This class will hold 2 class distribution, one for class 0 and one for class 1, and will predicit and instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
        
        Input
            - An instance to predict.
            
        Output
            - 0 if the posterior probability of class 0 is higher 1 otherwise.
        """
        return self.ccd0.class_value if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x) else self.ccd1.class_value 
    
def compute_accuracy(testset, map_classifier):
    """
    Compute the accuracy of a given a testset and using a map classifier object.
    
    Input
        - testset: The test for which to compute the accuracy (Numpy array).
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / #testset size
    """
    size = testset.shape[0]
    correct = 0.0
    for i in range(size):
        prediction = map_classifier.predict(testset[i])
        real_value = testset[i][-1]
        if prediction == real_value:
            correct += 1
    return (correct / size) * 100
    
            
            
            
            
            
            
            
            
            
    