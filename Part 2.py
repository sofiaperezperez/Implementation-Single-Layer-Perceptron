######################################################################
##############################   2 PART #############################
#####################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

## 1.2.1 EXPERIMENTAL SETUP ##

x = np.linspace(0, 5, 51)
y = (2*x**2)-11*x+1 

training = []
labels = []

for i in range(0, len(x), 8): # subsambpling
    training.append(x[i])
    labels.append(y[i] + np.random.normal(0,4,1)) # noise

training, labels = shuffle(training, labels, random_state=0)  # shuffle data to improve the training process
                
# Plot subsampled data with noisy traget values    
fig, ax=plt.subplots()
ax.plot(training, labels,".")
ax.set_title('')
ax.set_xlabel('x')
ax.set_ylabel('y')


##### 1.2.2 OPTIMIZATION: LMS-LEARNING RULE VS. CLOSED FORM #########  

# Transformation of data
D = 2 # Number of powers for the monomials
N = len(training)
expanded_data = np.zeros((N, D+1))

for d in range(D+1):      # making the design matrix
    expanded_data[:, d] = np.power(training, [d]*N)

## Weight vector using online LMS-learning rule ##
its = 0
obs = 0
maxIts = 150000 # Maximum number of iterations
w_online = np.random.normal(0, 1, D+1) # Initialized weight vector 
learning_rate = 0.00005

while its < maxIts : # Stopping criteria 

    if obs >= len(expanded_data):
        obs = 0   
    w_online +=  learning_rate * (labels[obs] - np.dot(expanded_data[obs], w_online)) * expanded_data[obs]
    obs +=1
    its +=1

y_predictions_online = (w_online[2]*x**2) + w_online[1]*x + w_online[0] # Predicted labels using the online trained weight vector w


## Weight vector using closed form ##
w_closed = np.linalg.inv(expanded_data.T @ expanded_data) @ expanded_data.T @ labels 
y_predictions_closed = (w_closed[2]*x**2) + w_closed[1]*x + w_closed[0]

# Comparison between the true function and the ones obtained with the different trained weight vectors:
plt.plot(training, labels,".")
plt.plot(x, y, 'r', label = 'True function')
plt.plot(x, y_predictions_online, 'g', label = 'Online predicted function')
plt.plot(x, y_predictions_closed, 'b', label = 'Closed predicted function')
plt.legend(loc = 'Upper left')
plt.show()



### 1.2.3 IMAGE DATA ###

# Creating the data:
n = 7
x = np.linspace(0, 5, 51)
y = (2*x**2)-11*x+1 
training = []
t = []

for i in range(0, len(x), 8): # subsambpling
    training.append(x[i])
    t.append(y[i])

data = [] # Matrix with 7 rows (one for each image) and 841 columns. 

for obs in range(n): 
    
    mu = 15
    sigma = 2
    m1 = np.random.normal(mu, sigma)
    m2 = np.random.normal(mu, sigma)
    img = np.ones((29,29)) # Image 29x29
    
    for i in range(1,29):
        for j in range(1,29):
            case1 = (i-m1)**2 +(j-m2)**2 -(3*training[obs])**2
            if case1 >= 0 :
                img[i,j] = 0
                
    data.append(img)

for element in range(len(data)):  
      data[element] = data[element].reshape(1, 841)[0] # Transformation of each 29x29 image into a 1x841 vector.


## Computation of w in closed form:
w_image = np.transpose(np.linalg.inv(data @ np.transpose(data) + 0.2 * np.identity(len(data))) @ data) @ np.transpose(t) # Choosing a small lambda as the matrix is not invertible.


## Plot the predicted labels vs the true ones for the 7 training images: 
predictions = []

for i in range(n):
    predictions.append(np.transpose(w_image) @ data[i])

plt.plot(training, predictions, '.', color = 'b', label = "predicted values")
plt.plot(training, t, '.', color = 'r', label = 'True values')
plt.legend(loc = 'Upper left')
plt.show()


## Error
def MSE(targets, preds):
    error = list(np.array(targets) - np.array(preds))
    squared_error = [i**2 for i in error]
    sum_squared_error = sum(squared_error)    
    loss = sum_squared_error/len(targets)
    return loss

error_subset = MSE(t, predictions)
print(error_subset) # 0.0016127928331364815


## Plot the predicted labels vs the true ones for the 51 images: 
n = 51
whole_data = [] # We repeat the process of creating the images for each of the 51 points

for obs in range(n): 
    
    mu = 15
    sigma = 2
    m1 = np.random.normal(mu, sigma)
    m2 = np.random.normal(mu, sigma)
    img = np.ones((29,29)) #Image 29x29 matrix
    
    for i in range(1,29):
        for j in range(1,29):
            case1 = (i-m1)**2 +(j-m2)**2 -(3*x[obs])**2
            if case1 >= 0 :
                img[i,j] = 0
                
    whole_data.append(img)


for element in range(len(whole_data)):      
    whole_data[element] = whole_data[element].reshape(1, 841)[0]


## Plotting:
predictions_whole_data = []

for i in range(n):
    predictions_whole_data.append(np.transpose(w_image) @ whole_data[i])
    
plt.plot(x, predictions_whole_data, '.', color = 'b', label = "predicted values")
plt.plot(x, y, '.', color = 'r', label = 'True values')
plt.legend(loc = 'Upper left')
plt.show()

error_whole_set = MSE(y, predictions_whole_data)
print(error_whole_set) # 6.032646220251331


## Increase noise variance for the centers:
# Creating the data:
n = 7
x = np.linspace(0, 5, 51)
y = (2*x**2)-11*x+1 
training = []
t = []

for i in range(0, len(x), 8): # subsambpling
    training.append(x[i])
    t.append(y[i])

data = [] # Matrix with 7 rows (one for each image) and 841 columns. 

for obs in range(n): 
    
    mu = 15
    sigma = 15 #it is increased (in the first experiment was 8)
    m1 = np.random.normal(mu, sigma)
    m2 = np.random.normal(mu, sigma)
    img = np.ones((29,29)) # Image 29x29
    
    for i in range(1,29):
        for j in range(1,29):
            case1 = (i-m1)**2 +(j-m2)**2 -(3*training[obs])**2
            if case1 >= 0:
                img[i,j] = 0
                
    data.append(img)

for element in range(len(data)):  
      data[element] = data[element].reshape(1, 841)[0] # Transformation of each 29x29 image into a 1x841 vector.


## Computation of w in closed form:
w_image = np.transpose(np.linalg.inv(data @ np.transpose(data) + 0.2 * np.identity(len(data))) @ data) @ np.transpose(t) # Choosing a small lambda as the matrix is not invertible.


## Plot the predicted labels vs the true ones for the 7 training images: 
predictions = []

for i in range(n):
    predictions.append(np.transpose(w_image) @ data[i])

plt.plot(training, predictions, '.', color = 'b', label = "predicted values")
plt.plot(training, t, '.', color = 'r', label = 'True values')
plt.legend(loc = 'Upper left')
plt.title('Predicted values vs the true ones with 7 images')
plt.show()


## Plot the predicted labels vs the true ones for the 51 images: 
n = 51
whole_data = [] # We repeat the process of creating the images for each of the 51 points

for obs in range(n): 
    
    mu = 15
    sigma = 15  #it is increased (in the first experiment was 8)
    m1 = np.random.normal(mu, sigma)
    m2 = np.random.normal(mu, sigma)
    img = np.ones((29,29)) #Image 29x29 matrix
    
    for i in range(1,29):
        for j in range(1,29):
            case1 = (i-m1)**2 +(j-m2)**2 -(3*x[obs])**2
            if case1 >= 0 :
                img[i,j] = 0
                
    whole_data.append(img)


for element in range(len(whole_data)):      
    whole_data[element] = whole_data[element].reshape(1, 841)[0]


## Plotting:
predictions_whole_data = []

for i in range(n):
    predictions_whole_data.append(np.transpose(w_image) @ whole_data[i])
    
plt.plot(x, predictions_whole_data, '.', color = 'b', label = "predicted values")
plt.plot(x, y, '.', color = 'r', label = 'True values')
plt.legend(loc = 'Upper left')
plt.title('Predicted values vs the true ones with 51 images')
plt.show()

### Showing the circles with the random center for sigma=2 and then increasing the sigma of a random variable to sigma=15 ######
 
 # In the first case we just plot the circles as we have it in assignment and in the second part we increased the sigma of the noise in the center of the circles 
 #for this case we decided to change the sigma= 15 and plot all the 7 pictures for train values of x . 
 # as we see, by increasing the noise the center is jumping out of our pictures sometimes. 
 
 # case 1
 # # sigma=2
 # case2    
 # Sigma= 15  
a=data[0]
im=a.reshape(29,29)
fig1 =plt.figure("""first 1""")
plt.gray()
plt.imshow(im)
plt.title('a')


fig2=plt.figure("""first 2""")
b=data[1]
im=b.reshape(29,29)
plt.gray()
plt.imshow(im)
plt.title('b')

fig3= plt.figure("""first 3""")
c=data[2]
im=c.reshape(29,29)
plt.gray()
plt.imshow(im)
plt.title('c')

fig4=plt.figure("""first 4""")
d=data[3]
im=d.reshape(29,29)
plt.gray()
plt.imshow(im)
plt.title('d')

fig5= plt.figure("""first 5""")
e=data[4]
im=e.reshape(29,29)
plt.gray()
plt.imshow(im)
plt.title('e')

fig6= plt.figure("""first 6""")
f=data[5]
im=f.reshape(29,29)
plt.gray()
plt.imshow(im)
plt.title('f')


fig7= plt.figure("""first 7""")
d=data[6]
im=d.reshape(29,29)
plt.gray()
plt.imshow(im)
plt.title('g')

plt.show()
