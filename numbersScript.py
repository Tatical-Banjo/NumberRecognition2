import numpy as np

import matplotlib.pyplot as plt

from mnist import MNIST
# Import `train_test_split`
from sklearn import svm
from sklearn import cluster
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.preprocessing import scale
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from sklearn.grid_search import GridSearchCV

def main():
	
	# Load in the data
	mndata = MNIST('testData')
	images_train, labels_train = mndata.load_training()
	images_test, labels_test  = mndata.load_testing()

	# Check distribution of numbers in training sample
	print('Digits:  0 1 2 3 4 5 6 7 8 9')
	print('labels: %s' % np.unique(labels_train))
	print('Class distribution: %s' % np.bincount(labels_train))

	
	
	# # # # # # # # # # # # # # # # # # # # # # # # # #
	# Format the data - 
	# # # # # # # # # # # # # # # # # # # # # # # # # #
	labels_train = np.array(labels_train)
	labels_test  = np.array(labels_test)
	# turn it from a 28^2 list into a 28x28 array
	for i in range(len(images_train)):
		images_train[i] = np.asfarray(images_train[i])

	for i in range(len(images_test)):
		images_test[i] = np.asfarray(images_test[i])

	
	# # The first thing that we’re going to do is preprocessing the data. You can standardize the digits data by, for example, making use of the scale() method:
	images_train = (images_train - np.min(images_train)) / np.ptp(images_train)
	images_test  = (images_test  - np.min(images_test))  / np.ptp(images_test)

	# Figure size (width, height) in inches
	fig = plt.figure(figsize=(6, 6))

	# Adjust the subplots 
	fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

	# For each of the 64 images
	for i in range(64):
	    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
	    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
	    # Display an image at the i-th position
	    ax.imshow(images_train[i].reshape(28,28), cmap=plt.cm.binary, interpolation='nearest')
	    # label the image with the target value
	    ax.text(0, 7, str(labels_train[i]))

	# plt.show()
	plt.savefig('numbersToLearn.png')
	plt.close() 
	

	# # Create a Randomized PCA model that takes two components
	# randomized_pca = RandomizedPCA(n_components=4)

	# # Fit and transform the data to the model
	# reduced_data_rpca = randomized_pca.fit_transform(images_train)

	# Create a regular PCA model 
	print('')
	print('Create a regular PCA model')
	pca = PCA(n_components=4)

	# Fit and transform the data to the model
	reduced_data_pca = pca.fit_transform(images_train)
	reduced_data_pca_test = pca.transform(images_test)

	# Inspect the shape
	# reduced_data_pca.shape

	# Print out the data
	print(reduced_data_pca.shape)
	print(reduced_data_pca)
	print(len(reduced_data_pca))
	print(type(reduced_data_pca))

	numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',]
	colors = ['red', 'blue', 'purple', 'yellow', 'white', 'black', 'lime', 'cyan', 'orange', 'gray']
	# for i in range(1000):
	# 	if(i%1000==0):
	# 		print(i)
	# 	x = reduced_data_pca[i, 0]
	# 	y = reduced_data_pca[i, 1]
	# 	plt.scatter(x, y, c=colors[labels_train[i]]) 
	# plt.legend(numbers)#, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	# plt.xlabel('First Principal Component')
	# plt.ylabel('Second Principal Component')
	# plt.title("PCA Scatter Plot")
	# plt.savefig('First_.pSeco_PrincipalComponentndng_ScatterPlot.png')
	# plt.close() 

	# for i in range(1000):
	# 	if(i%1000==0):
	# 		print(i)
	# 	x = reduced_data_pca[i, 0]
	# 	y = reduced_data_pca[i, 2]
	# 	plt.scatter(x, y, c=colors[labels_train[i]]) 
	# plt.legend(numbers)#, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	# plt.xlabel('Second Principal Component')
	# plt.ylabel('Third Principal Component')
	# plt.title("PCA Scatter Plot")
	# plt.savefig('Second_Third_PrincipalComponent_ScatterPlot.png')
	# plt.close() 
	
	# for i in range(1000):
	# 	if(i%1000==0):
	# 		print(i)
	# 	x = reduced_data_pca[i, 0]
	# 	y = reduced_data_pca[i, 3]
	# 	plt.scatter(x, y, c=colors[labels_train[i]]) 
	# plt.legend(numbers)#, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	# plt.xlabel('First Principal Component')
	# plt.ylabel('Fourth Principal Component')
	# plt.title("PCA Scatter Plot")
	# plt.savefig('First_.pFour_PrincipalComponentthng_ScatterPlot.png')
	# plt.close() 

	# # # # # # # # # # # # # # # # # # # # # # # # # # #
	# # Fit the data with a different moddel 
	# # # # # # # # # # # # # # # # # # # # # # # # # # #

	print("")
	print("SVC Model")
	# How to find good paramaters for the SVC Model

	# Set the parameter candidates
	parameter_candidates = [
	  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
	  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
	]

	# Create a classifier with the parameter candidates
	clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)

	# Train the classifier on training data
	print(reduced_data_pca)
	print(type(reduced_data_pca))
	print(reduced_data_pca.size)
	print(len(reduced_data_pca))



	# clf.fit(reduced_data_pca[1:5000][:], labels_train[1:5000]) 
	# # Print out the results 
	# print('Best score for training data:', clf.best_score_)
	# print('Best `C`:',clf.best_estimator_.C) #10
	# print('Best kernel:',clf.best_estimator_.kernel) #rfb
	# print('Best `gamma`:',clf.best_estimator_.gamma) #gamma
	# Best score for training data: 0.6399279855971194
	# Best `C`: 1000
	# Best kernel: rbf
	# Best `gamma`: 0.001

	nToTrain = 20000
	ntoPredict = 25

	# Train and score a new classifier with the grid search parameters
	svc_model = svm.SVC(C=1000, kernel='rbf', gamma=0.001).fit(reduced_data_pca[1:nToTrain][:], labels_train[1:nToTrain])

	# Assign the predicted values to `predicted`
	predicted = svc_model.predict(reduced_data_pca_test[1:ntoPredict][:]) 

	# Zip together the `images_test` and `predicted` values in `images_and_predictions`
	images_and_predictions = list(zip(images_test[1:ntoPredict][:], predicted))

	# # For the first 4 elements in `images_and_predictions`
	for index, (image, prediction) in enumerate(images_and_predictions):
	    # Initialize subplots in a grid of 1 by 4 at positions i+1
	    plt.subplot(5, 5, index + 1)
	    # Don't show axes
	    plt.axis('off')
	    # Display images in all subplots in the grid
	    plt.imshow(image.reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
	    # Add a title to the plot
	    plt.title('Predicted: ' + str(prediction))

	# # Show the plot
	plt.savefig('first4ElementsIn_images_and_predictions.png')
	plt.show()
	plt.close()

	print(labels_test[1:ntoPredict]) 
	print(predicted) 

	# Print the classification report of `y_test` and `predicted`
	print(metrics.classification_report(labels_test[1:ntoPredict], predicted))

	# for i in range(ntoPredict):
	# 	print('Truth = ',labels_test[i]," Predicted = ", predicted[i])

	# Print the confusion matrix
	print(metrics.confusion_matrix(labels_test[1:ntoPredict], predicted))

	# # Create an isomap and fit the `digits` data to it
	# X_iso = Isomap(n_neighbors=10).fit_transform(X_train)

	# # Compute cluster centers and predict cluster index for each sample
	# predicted = svc_model.predict(X_train)


   

if __name__ == "__main__":
    main()