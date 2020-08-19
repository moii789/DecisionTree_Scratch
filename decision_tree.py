#DECISION TREE CODE

def train_test_split_dt(df, test_size):
	indices = df.index.tolist();
	test_indices = random.sample(population = indices, k = test_size)
	test_df = df.loc[test_indices]
	training_df = df.drop(test_indices)
	return training_df, test_df

random.seed(0) 
training_df, test_df = train_test_split_dt(df, test_size)
test_df_svm = test_df.copy()
training_data = training_df.values

def check_purity(training_data):
	live_or_die_column = training_data[:, 0]
	unique_classes = np.unique(live_or_die_column)
	return len(unique_classes) == 1

def classify_data(training_data):
	live_or_die_column = training_data[:, 0]
	unique_classes, count_unique_classes = np.unique(live_or_die_column, return_counts = True)
	index = count_unique_classes.argmax()
	classification = unique_classes[index]
	return classification

def get_potential_splits(training_data):

	potential_splits = {}
	_, n_columns = training_data.shape
	for column_index in range(n_columns): 
		if column_index != 0:
			potential_splits[column_index] = []
			values_in_column = training_data[:, column_index]
			unique_values = np.unique(values_in_column)

			for index in range(len(unique_values)):
				if index != 0:
					current_value = unique_values[index]
					previous_value = unique_values[index - 1]
					potential_split = (current_value + previous_value) / 2
					potential_splits[column_index].append(potential_split)

	return potential_splits

potential_splits = get_potential_splits(training_data)

def split_data(training_data, split_column, split_value):
	split_column_vals = training_data[:, split_column]
	data_below = training_data[split_column_vals <= split_value]
	data_above = training_data[split_column_vals > split_value]

	return data_below,data_above

def calculate_entropy(training_data):
	label_column = training_data[:, 0]
	_, counts = np.unique(label_column, return_counts = True)
	probabilities = counts / counts.sum()
	entropy = sum(probabilities * -np.log2(probabilities))

	return entropy

def calculate_overall_entropy(data_below,data_above):
	n_data_points = len(data_below) + len(data_above)
	p_data_below = len(data_below) / n_data_points
	p_data_above = len(data_above) / n_data_points

	overall_entropy = p_data_below * calculate_entropy(data_below) + p_data_above * calculate_entropy(data_above)

	return overall_entropy

def calculate_best_split(training_data, potential_splits):
	
	overall_entropy = 999
	for column_index in potential_splits:
		for value in potential_splits[column_index]:
			data_below, data_above = split_data(training_data, column_index, value)
			current_entropy = calculate_overall_entropy(data_below,data_above)

			if current_entropy < overall_entropy:
				overall_entropy = current_entropy
				best_split_column = column_index
				best_split_value = value

	return best_split_column,best_split_value			

def main_algorithm(df, counter = 0):

	# preparing the data
	if counter == 0:
		data = df.values
	else:
		data = df

	#base case
	if check_purity(data):
		classification = classify_data(data)
		return classification	

	#recursive function
	else:
		counter += 1

		#helper functions 
		potential_splits = get_potential_splits(data)
		best_split_column, best_split_value = calculate_best_split(data, potential_splits)
		data_below, data_above = split_data(data, best_split_column,best_split_value) 	

		#instantiate sub-tree
		question = "{} <= {}".format(best_split_column, best_split_value)
		sub_tree = {question: []}

		#find answers
		yes_answer = main_algorithm(data_below, counter)
		no_answer = main_algorithm(data_above, counter)

		sub_tree[question].append(yes_answer)
		sub_tree[question].append(no_answer)

		return sub_tree

tree = main_algorithm(training_df)
			
def classify(example, tree):
	question = list(tree.keys())[0]
	feature_index, comparison_operator, value = question.split()

	#ask question
	if example[float(feature_index)] <= float(value):
		answer = tree[question][0]
	else:
		answer = tree[question][1]

	#base case
	if not isinstance(answer, dict):
		return answer
	else :
		return classify(example, answer)

def calculate_accuracy(test_df,tree):
	test_df[20] = test_df.apply(classify, axis = 1, args = (tree,))
	test_df[21] = test_df[20] == test_df[0]
	accuracy = test_df[21].mean()

	return accuracy

calculate_accuracy(test_df,tree)
actual = test_df[0]
predicted = test_df[20]
results_dt = confusion_matrix(actual, predicted) 
print ('Confusion Matrix Decision Tree:')
print(results_dt) 
print ('Accuracy Score Decision Tree:',accuracy_score(actual, predicted) * 100)
