import common


def part_one_classifier(data_train, data_test):
	# PUT YOUR CODE HERE
	# Access the training data using "data_train[i][j]"
	# Training data contains 3 cols per row: X in 
	# index 0, Y in index 1 and Class in index 2
	# Access the test data using "data_test[i][j]"
	# Test data contains 2 cols per row: X in 
	# index 0 and Y in index 1, and a blank space in index 2 
	# to be filled with class
	# The class value could be a 0 or a 1
	
        # Initialize weights
        weights = [0, 0, 0]  # 1 bias, two features
        updated_weights = True
        # repeat until weights haven't changed
        while updated_weights:
            updated_weights = False
            # Iterate through all training examples
            for i in range(common.constants.TRAINING_SIZE):
                # Get features vector (start with 1 for bias)
                features = [1, data_train[i][0], data_train[i][1]]
                
                # Make current prediction
                weighted_sum = 0
                for j in range(common.constants.NUM_FEATURES):
                    weighted_sum = weighted_sum + features[j]*weights[j]
                if weighted_sum > 0:
                    predicted_class = 1
                else:
                    predicted_class = 0

                # Compare to truth
                true_class = data_train[i][2]
                if predicted_class != true_class:
                    # If incorrect prediction, update weights
                    updated_weights = True
                    if true_class == 1:
                        delt = 1
                    else:
                        delt = -1
                    for j in range(common.constants.NUM_FEATURES):
                        weights[j] = weights[j] + delt * features[j]

        # after convergence, make our predictions on new data
        for i in range(common.constants.TEST_SIZE):
            # Get features vector (start with 1 for weights)
            features = [1, data_test[i][0], data_test[i][1]]
            
            # Make prediction
            weighted_sum = 0
            for j in range(common.constants.NUM_FEATURES):
                weighted_sum = weighted_sum + features[j]*weights[j]

            if weighted_sum > 0:
                predicted_class = 1
            else:
                predicted_class = 0
            
            # Update prediction
            data_test[i][2] = predicted_class
        return


def part_two_classifier(data_train, data_test):
	# PUT YOUR CODE HERE
	# Access the training data using "data_train[i][j]"
	# Training data contains 3 cols per row: X in 
	# index 0, Y in index 1 and Class in index 2
	# Access the test data using "data_test[i][j]"
	# Test data contains 2 cols per row: X in 
	# index 0 and Y in index 1, and a blank space in index 2 
	# to be filled with class
	# The class value could be a 0 or a 8
	
        # Initialize weights array to 0
        # weights[i] will access weights for digit i
        weights = [[0, 0] for i in range(common.constants.NUM_CLASSES)]
        
        updated_weights = True
        # repeat until weights haven't changed
        while updated_weights:
            updated_weights = False
            # Iterate through all training examples
            for i in range(common.constants.TRAINING_SIZE):
                # Get features vector (no bias)
                features = [data_train[i][0], data_train[i][1]]

                # calculate score for each digit (class)
                d_scores = [0 for i in range(common.constants.NUM_CLASSES)]
                for digit in range(common.constants.NUM_CLASSES):
                    d_score = 0
                    for j in range(2):
                        d_score = d_score + features[j]*weights[digit][j]
                    
                    # save score for digit
                    d_scores[digit] = d_score

                # prediction = max d_score
                predicted_class = d_scores.index(max(d_scores))

                # Check against truth
                true_class = int(data_train[i][2])

                # If incorrect prediction, update weights
                if predicted_class != true_class:
                    updated_weights = True
                    # Increase weights of true class
                    for j in range(2):
                        weights[true_class][j] += features[j]

                    # Decrease weights of wrong class
                    for j in range(2):
                        weights[predicted_class][j] -= features[j]

        # After convergence, make predictions
        for i in range(common.constants.TEST_SIZE):
            # Get features vector (no bias)
            features = [data_test[i][0], data_test[i][1]]

            # calculate score for each digit (class)
            d_scores = [0 for i in range(common.constants.NUM_CLASSES)]
            for digit in range(common.constants.NUM_CLASSES):
                d_score = 0
                for j in range(2):
                    d_score = d_score + features[j]*weights[digit][j]
                    
                # save score for digit
                d_scores[digit] = d_score

            # prediction = max d_score
            data_test[i][2] = d_scores.index(max(d_scores))
        
        return



