import numpy as np
np.random.seed(42)

chi_table = {0.01  : 6.635,
             0.005 : 7.879,
             0.001 : 10.828,
             0.0005 : 12.116,
             0.0001 : 15.140,
             0.00001: 19.511}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the dataset.    
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.#
    # for every different class find Si (occurences of class i in data S):
    # calculate probability (Si/S) squared, sum all and return 1 - (sum)
    unique, counts = np.unique(data[:,-1], return_counts=True)
    gini = 1 - np.sum(np.square(counts / data.shape[0]))
    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.    
    """
    entropy = 0.0
    
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    unique, counts = np.unique(data[:,-1], return_counts=True)
    entropy = np.abs(np.sum(np.array([(v/data.shape[0]) * np.log2(v/data.shape[0]) for v in counts])))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic 
    # functionality as described in the notebook. It is highly recommended that you 
    # first read and understand the entire exercise before diving into this class.
    
    def __init__(self, feature=None, value=None, data=None):
        self.children = []
        self.data = data
        self.feature = feature # column index of criteria being tested
        self.value = value # value necessary to get a true result
        
    def add_child(self, node):
        self.children.append(node)
        
    def has_children(self):
        return len(self.children) > 0 and (self.children[0] or self.children[1])


        
def avg_of_pairs(lis):
    if len(lis) < 2:
        return lis
    return np.array([(x+y)/2.0 for x,y in zip(lis, lis[1:])])

def find_best_attribute(data, impurity):
    c = data.shape[1] - 1
    node_impurity = impurity(data)
    best = 0
    attr = 0
    value = 0
    left = np.array([])
    right = np.array([])
    for i in range(c):
        thresholds = avg_of_pairs(data[:,i])
        for threshold in thresholds:
            left_child = data[data[:,i] <= threshold]
            right_child = data[data[:, i] > threshold]
            split = impurity(left_child) + impurity(right_child)
            gain = node_impurity - split
            if (gain > best):
                best = gain
                left = left_child
                right = right_child
                attr = i
                value = threshold
    return left,right,value,attr

def get_chi(left, right):
    chi = 0
    p0 = left[left[:, -1]<1].shape[0]
    n0 = left.shape[0] - p0
    D0 = left.shape[0]
    p1 = right[right[:,-1]<1].shape[0]
    n1 = right.shape[0] - p1
    D1 = right.shape[0]
    PY0 = (left[left[:,-1]<1].shape[0] + right[right[:,-1]<1].shape[0]) / (left.shape[0] + right.shape[0])
    PY1 = 1 - PY0
    E0l = D0 * PY0
    E1l = D0 * PY1
    
    E0r = D1 * PY0
    E1r = D1 * PY1
    l = (((p0 - E0l) ** 2)/E0l) + (((n0 - E1l) ** 2) / E1l)
    r = (((p1 - E0r) ** 2)/E0r) + (((n1 - E1r) ** 2) / E1r)
    chi = l + r
    return chi

def build_tree(data, impurity, chi_value=1):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    root = DecisionNode(data=data)
    
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    def add_nodes(node):
        if not node:
            return
        curr_data = node.data
        if not curr_data.any() or not impurity(data):
            return
        left,right,value,feature = find_best_attribute(curr_data, impurity)
        node.value = value
        node.feature = feature
        left_child = None
        right_child = None
        should_prune = False
        chi = 0
        if left.any():
            left_child = DecisionNode(data=left)
        if right.any():
            right_child = DecisionNode(data=right)
        if left_child and right_child and chi_value != 1:
            chi = get_chi(left, right)
            should_prune = chi < chi_table[chi_value]
        if not should_prune:
            add_nodes(left_child)
            add_nodes(right_child)
            node.add_child(left_child)
            node.add_child(right_child)
        else:
            print ('Pre-pruned', chi)
        
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    add_nodes(root)
    return root

    

def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    # TODO: Implement the function. 
    def traverse(node, instance):
        if not node:
            return
        if not node.has_children():
            return node.data[:,-1][0]
        feature = node.feature
        threshold = node.value
        if instance[feature] <= threshold:
            return traverse(node.children[0], instance)
        else:
            return traverse(node.children[1], instance)
    ###########################################################################
    pred = traverse(node, instance)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred

def calc_accuracy(node, dataset):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given dataset.

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0.0
    predictions = 0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for data in dataset:
        success = predict(node, data) == data[-1]
        if success:
            predictions += 1.0
    accuracy = predictions / dataset.shape[0]
    print ('Accurate predictions: {} out of {}'.format(predictions, dataset.shape[0]))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy

def print_tree(node):
    '''
    prints the tree according to the example in the notebook

	Input:
	- node: a node in the decision tree

	This function has no return value
	'''

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################    
    def recursion_print(node, i):
        if not node:
            return
        to_print = '[X{} <= {}]'.format(node.feature, node.value) if node.has_children() else 'leaf: [{}: {}]'.format(node.data[:,-1][0], node.data.shape[0])
        print ((' ' * i * 2) + to_print)
        if node.has_children():
            recursion_print(node.children[0], i+1)
            recursion_print(node.children[1] or None, i+1)
    recursion_print(node, 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
