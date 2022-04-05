### 1.1 PART1: BINARY CLASSIFICATION AND THE PERCEPTRON ###

### 1.1.1 THE MNIST DATA SET ###

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels

# Reading the data:
train, train_labels = load_mnist(r'')
test, test_labels = load_mnist(r'', kind = 't10k')


# Plot an image:
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils import shuffle 


# Reduction of the dataset, only two labels: 0 -> t-shirt; 5 -> sandals

# Train:
index_train = np.where((train_labels == 0) | (train_labels == 5))
labels_tr_small = train_labels[index_train]
labels_tr_small = labels_tr_small.astype(int)
train_small = train[index_train]

# Test:
index_test = np.where((test_labels == 0) | (test_labels == 5))
test_small = train[index_test]
labels_test_small = test_labels[index_test]
labels_test_small = labels_test_small.astype(int)


###############  FEATURE EXTRACTION  #############3
pca = PCA()
result_pca = pca.fit(train_small)
explained_variance = np.cumsum(result_pca.explained_variance_ratio_) # Cumulative variance explained by each component

# Plot cumulative variance:
plt.plot(explained_variance)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

# Plot cumulative variance of the first 10 components:
plt.scatter(np.arange(1,11), explained_variance[:10])
plt.xlabel('Principal components')
plt.ylabel('Cumulative explained variance')
plt.show()

# We keep the first two components, which explain more than 50% of the original variance:
principal_components = result_pca.components_[0:2]
new_data = train_small @ np.transpose(principal_components) # Transformed data.

# Plot data with 2 features and their corresponding classes:
plt.scatter(new_data[:,0], new_data[:,1], c=labels_tr_small, alpha=0.5)
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()



### 1.1.2 PERCEPTRON TRAAINING ALGORITHM ###

# Changing labels to 1 or -1 for the perceptron algorithm:
for i in range(len(labels_tr_small)):
    
    if labels_tr_small[i] == 0:        
        labels_tr_small[i] = -1          
    
    else: 
        labels_tr_small[i] = 1
    

## Homogeneous coordinates (we add '1' as the first component of each observation):
final_tr = []

for i in range(len(new_data)):
    result = [1]
    
    for j in range(len(new_data[i])):
        result.append(new_data[i][j])        
        
    final_tr.append(result)


## Implementation of perceptron functions: 
def perc(w, X):
    
    threshold = 0.0    
    prediction = []
    
    for observation in X:
        total = np.dot(observation, w)
        if total > threshold:
            prediction.append(1.0)
            
        else:
            prediction.append(-1.0)
            
    return prediction
    

def percTrain(X, t, maxIts, online = True):
    
        learning_rate = 1
        counter = 0
        w = np.random.normal(size = len(X[0])) # Initialize w as a Gaussian vector.
        
        if online: # Online training
            
            while counter < maxIts:  # Stopping criteria  
                
                shuffled_X, shuffled_t = shuffle(X, t, random_state=0) # Shuffle data to pick a random observation that has been wrongly predicted                    
                prediction = perc(w, shuffled_X)
                obs = 0
                
                while prediction[obs] == shuffled_t[obs]: # search for random observation with different label from prediction
                    obs += 1
                    if obs >= len(shuffled_t): # All observations have been correctly predicted
                        return w              
                
                for index in range(len(w)):
                    w[index] = w[index] + learning_rate * (shuffled_t[obs]) * shuffled_X[obs][index]
                    
                counter += 1  
        
        if not online: # batch version
            
            n_epoch = maxIts
            
            for i in range(n_epoch):
                
                prediction = perc(w, X)
                labs = np.zeros(len(prediction)) # Contains '0' if the observation is correctly predicted and its label otherwise.
                
                for i in range(len(labs)):
                    if prediction[i] == t[i]:
                        labs[i] = 0
                    else:
                        labs[i] = t[i]
                
                for index in range(len(w)):                    
                    w[index] = w[index] + learning_rate * np.dot(np.transpose(X)[:][index], labs) # Each component of w is updated if that observation has been worngly predicted     
        
        return w
    

def error_rate(weights, x, labels):    
    return (1 - accuracy(weights, x, labels))

def accuracy(weights, x, labels):
    
    prediction = perc(weights, x)        
    accuracy = 0
    
    for j in range(len(labels)):
        if labels[j] == prediction[j]:
            accuracy +=1
    
    accuracy = float(accuracy)/float(len(x))
    
    return accuracy
                    
 
# Training the perceptron using the dataset
w_batch = percTrain(final_tr, labels_tr_small, 100, 0)  # error rate on training set: 0.010583333333333278
w_online = percTrain(final_tr, labels_tr_small, 100, 1) # error rate on training set: 0.012416666666666631

# Line for plotting the decision boundary       
def line(x, slope, intercept):
    return slope*x + intercept 

# Equations for the slope and intercept:
slope_w_batch = -(w_batch[0]/w_batch[2])/(w_batch[0]/w_batch[1])  
intercept_w_batch = -w_batch[0]/w_batch[2]
slope_w_online = -(w_online[0]/w_online[2])/(w_online[0]/w_online[1])  
intercept_w_online = -w_online[0]/w_online[2]

# Plotting the data and the decision boundary obtained with the batch version weight vector:     
plt.scatter(new_data[:,0], new_data[:,1], c=labels_tr_small, alpha=0.5)
plt.plot(new_data[:,0], line(new_data[:,0], slope_w_batch, intercept_w_batch), 'g.-')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Batch version')
plt.show()  

# Plotting the data and the decision boundary obtained with the online version weight vector: 
plt.scatter(new_data[:,0], new_data[:,1], c=labels_tr_small, alpha=0.5)
plt.plot(new_data[:,0], line(new_data[:,0], slope_w_online, intercept_w_online), 'g.-')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Online version')
plt.show()                
                        


### FEATURE TRANSFORMATION (6 FEATURES) ###

feature_transformation = []

for i in range(len(final_tr)): # Using homogeneous coordinates
    feature_transformation.append([1, final_tr[i][1], final_tr[i][2], final_tr[i][1]**2, final_tr[i][2]**2, final_tr[i][1] * final_tr[i][2]])    

# Training the perceptron using the transformed dataset
w_transf_batch = percTrain(feature_transformation, labels_tr_small, 100, 0) # Error rate on training set: 0.010916666666666686
w_transf_online = percTrain(feature_transformation, labels_tr_small, 100, 1) # Error rate on training set: 0.3396666666666667


# Sample of the relevant region using meshgrid:
h = 2 # step size in the mesh
x_min, x_max = 0, 3000 
y_min, y_max = 0, 3000
x, y = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Z = np.transpose(w_transf_batch)*feature_transformation
Z = w_transf_batch[0] + w_transf_batch[1] * x + w_transf_batch[2] * y + w_transf_batch[3] * (x**2) + w_transf_batch[4] * (y**2) + w_transf_batch[5] * x * y

tr_1 = [] # Relevant data
labels_1 = [] # Corresponding labels
for i in range(len(new_data)):
    if new_data[i,1] <3000 and new_data[i,0] < 3000:
        new_element  = []
        for j in range(len(new_data[i])):
            new_element.append(new_data[i,j])
        tr_1.append(new_element)
        labels_1.append(labels_tr_small[i])
tr_1 = [np.transpose(tr_1[:])[0], np.transpose(tr_1[:])[1]]      
    

# Plot the contour function and the data in 2D:
plt.contour(x, y, Z)
plt.scatter(tr_1[0], tr_1[1], c=labels_1, cmap=plt.cm.Paired)
plt.show()



####### TRAINING USING ALL PIXELS (785 FEATURES)########

pixels = []

for i in range(len(train_small)): # Homogeneous coordinates
    result = [1]
    
    for j in range(len(train_small[i])):
        result.append(train_small[i][j])        
        
    pixels.append(result)

w_pixels_online = percTrain(pixels, labels_tr_small, 100, 1) # Error rate on training set: 0.003249999999999975


### PLOT THE WEIGHTS AFTER TRAINING (online) ###
item = w_pixels_online.copy()
item = list(item)
del item[0]
item_image = np.array(item).reshape(28, 28)
plt.imshow(item_image, cmap = "gray")
plt.axis("off")
plt.show()


### ERROR RATE ###

# Re-labeling the labels of the target values for the perceptron algorithm:
for i in range(len(labels_test_small)):    
    if labels_test_small[i] == 0:
        
        labels_test_small[i] = -1          
    
    else: 
        labels_test_small[i] = 1

# Projection of the test set into the two principal components:
new_test_data = test_small @ np.transpose(principal_components) # PCA
final_test = [] # Homogeneous coordinates

for i in range(len(new_test_data)):
    result = [1]    
    for j in range(len(new_test_data[i])):
        result.append(new_test_data[i][j])
                
    final_test.append(result)


# Error rate on test set with 2 features:
error_w_batch = error_rate(w_batch, final_test, labels_test_small)  
error_w_online = error_rate(w_online, final_test, labels_test_small) 
print('Error batch:', error_w_batch)   # Error rate on test set: 0.502
print('Error online:', error_w_online) # Error rate on test set: 0.5


# Error rate on test set with 5 features:
feature_transformation_test = []

for i in range(len(final_test)):
    feature_transformation_test.append([1, final_test[i][1], final_test[i][2], final_test[i][1]**2, final_test[i][2]**2, final_test[i][1] * final_test[i][2]])
    
error_w_transf_batch = error_rate(w_transf_batch, feature_transformation_test, labels_test_small)  
error_w_transf_online = error_rate(w_transf_online, feature_transformation_test, labels_test_small) 
print('Error batch:', error_w_transf_batch)   # Error rate on test set: 0.5015000000000001
print('Error online:', error_w_transf_online) # Error rate on test set: 0.501

# Error rate on test set with 784 features:
pixels_test = []

for i in range(len(test_small)):
    result = [1]
    
    for j in range(len(test_small[i])):
        result.append(test_small[i][j])
        
        
    pixels_test.append(result)


error_pix_online = error_rate(w_pixels_online, pixels_test, labels_test_small) 
print('Error online:', error_pix_online) # Error rate on test set: 0.49950000000000006