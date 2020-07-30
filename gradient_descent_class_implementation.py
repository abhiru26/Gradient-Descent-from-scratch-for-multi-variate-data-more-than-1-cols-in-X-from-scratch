class Gradient_descent:


    '''
    This is a class that helps to analyse the Gradient Descent Values for AI. It has two methods, One for gredient descent values that return as a dictionary and the Second one that predict the values of the dataset. For this two methods, required to create dataset when using the methods.


    Author ----> Sayan Roy
    Date ------> 27 July, 2020
    '''
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn import datasets

    


    def gradient_descent_function(self, no_features = 3, no_samples = 100, test_size_value = 0.2, random_state_value = 1, epochs_value = 200, learning_rate = 0.001):

        '''
        This methos is for analysing the Gradient Descent Values of created dataset. It creates the dataset and return the Gradient Descent Values values as a Dictionary. In the dictionary, the key are the iteration values and the key-values are the gradient descent values of that iterations.


        Arguments for creation a Dataset ------------->

        no_features ---> how many features you want to create. eg. no_features = 3
        no_samples ----> how many samples/rows you want for the dataset. eg. no_samples = 100




        Arguments to train the model based on the created dataset ------------->

        test_size_value ----> how many datas of your datset want for testing the trained model. For 20%, input 0.2 eg. test_size_value = 0.2 

        random_state_value ----> for split the dataset according to the test_size value. eg. random_state_value = 1



        Required arguments for creating the  the dictionary of the gradient descent ------------->

        epochs_value -----> How many iterations you want for the gradient descent. eg. epochs_value = 200
        learning_rate -----> The value of the learning rate for gradient descent. eg. learning_rate = 0.001



        Author ----> Sayan Roy
        Date -----> 27 July, 2020

        '''
        
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn import datasets
        X, y = datasets.make_regression(n_samples = no_samples, n_features = no_features, n_informative = 1, noise = 30, random_state = 0)

        # gradient_descent(no_features, testSize, randomState, X, y)
        



        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size_value, random_state = random_state_value)

        from sklearn.linear_model import LinearRegression
        LR = LinearRegression()

        LR.fit(X_train, y_train)

        # print("The values of X_train ---->", X_train)

        constant_list = []
        trained_features = []

        for i in range(no_features):
            j = "w" + str(i)
            constant_list.append(j)

            feature = X_train[:, i].reshape(len(X_train))
            # feature = X_train[:, i]
            # print(feature)
            trained_features.append(feature)




        epochs = epochs_value
        lr = learning_rate
        c = 0

        parameters = []


        result_dict = {}

        for iteration in range(epochs):

            parameters = []

            if iteration == 0:
                for i in range(no_features):
                    constant = constant_list[i] = 0
                    k = constant * trained_features[i]
                    parameters.append(k)
            else:

                derivative_variable_values

                for i in range(no_features):
                    constant = derivative_variable_values[i]
                    k = constant * trained_features[i]
                    parameters.append(k)

            y_pred = c
            for i in parameters:
                y_pred += i

            loss = sum(y_train - y_pred) ** 2 / len(X_train)

            d_c = (-2 / len(X_train)) * sum(y_train - y_pred)   

            c = c - (lr * d_c)
            derivative_variable = []

            for i in range(no_features):
                i = str(i)

                variable = "d_" + i
                derivative_variable.append(variable)

            print("derivative_variable---->",derivative_variable)
            derivative_variable_values = []
            for j in range(no_features):
                # print(derivative_variable[1]) 
                # print(trained_features[1])


                value = derivative_variable[j] = (-2 / len(X_train)) * sum((y_train - y_pred) * trained_features[j])
                # print(value)
                value_constants = constant_list[j] = constant_list[j] - (lr * value)

                derivative_variable_values.append(value_constants)


                result_dict[iteration] = loss

        return result_dict
        # print(len(result_dict))

        # print("The loss after ", i , "iteration is ", loss)


        # result_dict

    def predict_value(self, no_features = 3, no_samples = 100, test_size_value = 0.2, random_state_value = 1, predicted_values = [[1,1,1]]):


        '''
        This method is for prediction. Create a dataset by giving the values and also give it the values that have to predict. Then this methos predicts and return the value of that prediction.

        Arguments for creation a Dataset ------------->

        no_features ---> how many features you want to create. eg. no_features = 3
        no_samples ----> how many samples/rows you want for the dataset. eg. no_samples = 100





        Arguments to train the model based on the created dataset ------------->

        test_size_value ----> how many datas of your datset want for testing the trained model. For 20%, input 0.2 eg. test_size_value = 0.2 

        random_state_value ----> for split the dataset according to the test_size value. eg. random_state_value = 1





        Arguments for prediction based on the created dataset ------------->

        predicted_values ----> It takes 2-D array. eg. predicted_values = [[1,1,1]]


        Author ----> Sayan Roy

        Date -----> 27 July, 2020


        '''

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn import datasets


        X, y = datasets.make_regression(n_samples = no_samples, n_features = no_features, n_informative = 1, noise = 30, random_state = 0)



        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size_value, random_state = random_state_value)


        from sklearn.linear_model import LinearRegression
        LR = LinearRegression()

        LR.fit(X_train, y_train)

        return LR.predict(predicted_values)



