import numpy as np
from utils.features import prepare_for_training
class LinearRegression:
    def __init__(self, data, labels,polynomial_degree=0,sinusoid_degree=0, normalized_data=True):
        """
        data preprocessed
        get the number of features
        initilize feature matrix
        """
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalized_data = normalized_data
        (data_processed,features_mean,features_deviation)=prepare_for_training(data, polynomial_degree, sinusoid_degree, normalized_data)
        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features,1))

    def train(self, learning_rate, num_iterations=500):
        cost_hist = self.gradient_descent(learning_rate, num_iterations)
        return self.theta, cost_hist

    
    def gradient_descent(self, learning_rate, num_iterations=500):
        cost_hist = []
        for _ in range(num_iterations):
            self.gradient_step(learning_rate)
            cost_hist.append(self.cost_function(self.data, self.labels))
        return cost_hist

    def gradient_step(self, learning_rate):
        """
        update theta using matrix operation
        """
        num_examples = self.data.shape[0]
        prediction = LinearRegression.prediction_method(self.data, self.theta)
        delta = prediction - self.labels
        theta = self.theta
        theta = theta - learning_rate * (1/num_examples) * (np.dot(delta.T,self.data))
        self.theta = theta.T


    def cost_function(self, data, labels):
        num_examples = data.shape[0]
        predictions = LinearRegression.prediction_method(data, self.theta)
        delta = predictions - labels
        cost = (1/2) * np.dot(delta.T, delta)
        return cost[0][0]



    @staticmethod
    def prediction_method(data, theta):
        predictions = np.dot(data, theta)
        return predictions
    
    def get_cost(self, data, labels):
        data_processed=prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalized_data=True)[0]
        return self.cost_function(data_processed, labels)
    
    def predict(self,data):
        data_processed=prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalized_data=True)[0]
        predictions = LinearRegression.prediction_method(data_processed, self.theta)
 

