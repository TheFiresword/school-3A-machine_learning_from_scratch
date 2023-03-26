# %% [markdown]
# ## Descent-gradient algorithm implementation following:
#     Batch method
#     On-line method
# ### And using different loss functions:<br>
#     L2 error
#     Mean Square Error

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
def changment(loss_function, weights, label, features):
    if loss_function == "quadratic":
        factor = -2*(label - weights.T.dot(features))
        return factor*features 
    elif loss_function == "l1":
        pass

# %%
def sum_changment(loss_function, weights, all_labels, all_features):
    if loss_function in {"quadratic", "MSE"}:
        delta = np.zeros(weights.shape)
        for i in range(len(all_labels)):
            delta += changment("quadratic", weights, all_labels[i], all_features[i])
        #print(f"Delta is {delta}")
    return delta if loss_function == "quadratic" else delta/len(all_labels)

# %%

def descent_gradient(loss_function:str, labels:np.array, record_features:np.array, nepochs:int, learning_rate:float, method="batch" or "on-line"):
    size = len(labels)

    init_shape = record_features.shape

    nweights = 1 if len(init_shape) == 1 else init_shape[1]
    weights = np.random.rand(nweights+1)

    # Adding the feature 1 for the bias node
    bias_feature = np.ones(size)

    cpy_record_features = record_features.reshape((size, 1)) if nweights == 1 else record_features
    cpy_record_features = np.hstack((cpy_record_features, bias_feature.reshape((size, 1))))
    #print(cpy_record_features, cpy_record_features[0])

    if method == "on-line":
        for _ in range(nepochs):
            for i in range(size):
                weights = weights - learning_rate*changment(loss_function, weights, labels[i], cpy_record_features[i])
    
    elif method == "batch":
        for _ in range(nepochs):
            weights = weights - learning_rate*sum_changment(loss_function, weights, labels, cpy_record_features)

    return weights

# %%
loss_function = "quadratic"
nepochs = 10000
learning_rate = 0.01
labels = np.array([3.5, 5.2, 4.5, 5.3, 6.5, 4.1])
features_X= np.array([0.1, 0.2, 0.3, 0.3, 0.5, 0.4])

# %%
w = descent_gradient(loss_function, labels, features_X, nepochs, learning_rate, "batch")


# %%
plt.plot(features_X, labels, 'ro', label="Original data")
z = np.linspace(0, 0.6, 100)
plt.plot(z, w[0]*z+w[-1], 'b', label="Original data")
plt.show()

# %%
multdm_features = np.array([
    [0.5, 3],
    [0.4, 3],
    [0.4, 4],
    [2.3, 5],
    [2.1, 5],
    [2.2, 4.5]
])

_w_ = descent_gradient("quadratic", labels, multdm_features, 10000, 0.001, "batch")
print(_w_)


# %%
plt.plot(multdm_features[:, 0], multdm_features[:, 1], 'ro', label="Original data")
plt.show()

# %%
labels = np.array([1, 1,  1, -1, -1, -1])

def perceptron(cpy_multdm_features, labels, nepochs, learning_rate):

    init_shape = cpy_multdm_features.shape
    nweights = 1 if len(init_shape) == 1 else init_shape[1] -1
    weights = np.random.rand(nweights+1)
    

    for _ in range(nepochs):
        misclassified = 0
        delta = 0
        for i in range(len(labels)):
            cost = labels[i]*weights.T.dot(cpy_multdm_features[i])
            if cost <= 0:
                misclassified += 1
                delta += labels[i]*cpy_multdm_features[i]
        weights = weights + learning_rate*delta
    return weights

cpy_multdm_features = np.hstack((multdm_features, np.ones(len(labels)).reshape((-1, 1))))
final = perceptron(multdm_features, labels, 100, 0.1)
#print(final)
posCategorie = np.array([(multdm_features[:, 0][i], multdm_features[:, 1][i]) for i in range(len(labels)) if labels[i] == 1])
negCategorie = np.array([(multdm_features[:, 0][i], multdm_features[:, 1][i]) for i in range(len(labels)) if labels[i] == -1])

plt.plot(posCategorie[:, 0], posCategorie[:, 1], '+')
plt.plot(negCategorie[:, 0], negCategorie[:, 1], 'o')

z = np.linspace(0, 10, 10)
separator_equation = (-final[0]*z-final[-1])/final[1]
plt.plot(z, separator_equation, 'b')
plt.show()

# %%
import pandas as pd
iris_data = pd.read_csv("C:/Users/Junior/Documents/3A/AI/TDs/iris_a.txt", header=None)
feature_array = np.array(iris_data)
size = len(feature_array)

# %%

posCategorie = np.array([(feature_array[:, 0][i], feature_array[:, 1][i]) for i in range(size) if feature_array[:, -1][i] == 1])
negCategorie = np.array([(feature_array[:, 0][i], feature_array[:, 1][i]) for i in range(size) if feature_array[:, -1][i] == -1])
e = plt.figure(0)
plt.plot(posCategorie[:, 0], posCategorie[:, 1], '+')
plt.plot(negCategorie[:, 0], negCategorie[:, 1], 'o')
e.show()


posCategorie = np.array([(feature_array[:, 0][i], feature_array[:, 2][i]) for i in range(size) if feature_array[:, -1][i] == 1])
negCategorie = np.array([(feature_array[:, 0][i], feature_array[:, 2][i]) for i in range(size) if feature_array[:, -1][i] == -1])

f = plt.figure(1)
plt.plot(posCategorie[:, 0], posCategorie[:, 1], '+')
plt.plot(negCategorie[:, 0], negCategorie[:, 1], 'o')
f.show()

g = plt.figure(2)
posCategorie = np.array([(feature_array[:, 0][i], feature_array[:, 3][i]) for i in range(size) if feature_array[:, -1][i] == 1])
negCategorie = np.array([(feature_array[:, 0][i], feature_array[:, 3][i]) for i in range(size) if feature_array[:, -1][i] == -1])

plt.plot(posCategorie[:, 0], posCategorie[:, 1], '+')
plt.plot(negCategorie[:, 0], negCategorie[:, 1], 'o')
g.show()

h = plt.figure(3)
posCategorie = np.array([(feature_array[:, 1][i], feature_array[:, 2][i]) for i in range(size) if feature_array[:, -1][i] == 1])
negCategorie = np.array([(feature_array[:, 1][i], feature_array[:, 2][i]) for i in range(size) if feature_array[:, -1][i] == -1])

plt.plot(posCategorie[:, 0], posCategorie[:, 1], '+')
plt.plot(negCategorie[:, 0], negCategorie[:, 1], 'o')
h.show()

i = plt.figure(4)
posCategorie = np.array([(feature_array[:, 1][i], feature_array[:, 3][i]) for i in range(size) if feature_array[:, -1][i] == 1])
negCategorie = np.array([(feature_array[:, 1][i], feature_array[:, 3][i]) for i in range(size) if feature_array[:, -1][i] == -1])

plt.plot(posCategorie[:, 0], posCategorie[:, 1], '+')
plt.plot(negCategorie[:, 0], negCategorie[:, 1], 'o')
i.show()

j = plt.figure(5)
posCategorie = np.array([(feature_array[:, 1][i], feature_array[:, 3][i]) for i in range(size) if feature_array[:, -1][i] == 1])
negCategorie = np.array([(feature_array[:, 1][i], feature_array[:, 3][i]) for i in range(size) if feature_array[:, -1][i] == -1])

plt.plot(posCategorie[:, 0], posCategorie[:, 1], '+')
plt.plot(negCategorie[:, 0], negCategorie[:, 1], 'o')
j.show()

# %% [markdown]
# ## Training of a Perceptron model on iris dataset

# %%
#print(feature_array[:, 0:2], feature_array[:, -1])

np.random.shuffle(feature_array)
#print(feature_array)
labels = feature_array[:, -1]
cpy_multdm_features = np.hstack((feature_array[:, 0:2], np.ones(len(labels)).reshape((-1, 1))))

# %%

"""
#------------------------------------------------------------#
Splitting data in train set and test set
"""
#print(cpy_multdm_features[0:5,:])

fifty = len(labels)//2
train_set = cpy_multdm_features[0:fifty, :]
train_y = labels[0:fifty]

test_set = cpy_multdm_features[fifty: , :]
test_y = labels[fifty:]



#------------------------------------------------------------#
#Training the perceptron

iris_weights = perceptron(train_set, train_y, nepochs, learning_rate)

print(iris_weights)
posCategorie = np.array([(feature_array[0:fifty, 0][i], feature_array[0:fifty, 1][i]) for i in range(len(train_y)) if feature_array[0:fifty, -1][i] == 1])
negCategorie = np.array([(feature_array[0:fifty, 0][i], feature_array[0:fifty, 1][i]) for i in range(len(train_y)) if feature_array[0:fifty, -1][i] == -1])

#print(posCategorie)
#print(negCategorie)
plt.plot(posCategorie[:, 0], posCategorie[:, 1], '+')
plt.plot(negCategorie[:, 0], negCategorie[:, 1], 'o')

z = np.linspace(0, 10, 10)
separator_equation = (-iris_weights[0]*z-iris_weights[-1])/iris_weights[1]
plt.plot(z, separator_equation, 'b')
plt.show()




# %%
#------------------------------------------------------------#
#Testing the perceptron

def predict(weights:np.array, dim_size, test_set, test_y):
    weights.dot()
    pass

predict(iris_weights, len(iris_weights), test_set)


