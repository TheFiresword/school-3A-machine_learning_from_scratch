import numpy as np
import pandas as p
import matplotlib.pyplot as plt

test = p.read_csv("data.txt")
value = np.asarray(test)

f = open("data.txt","r")
info = f.read()
text_data = info.split('\n')
data = []
for x in range(len(text_data)):
   line = text_data[x].split(',')
   data.append([])
   for i in range(len(line)):
       if line[i] != '':
        data[x].append(float(line[i]))
if data[len(data)-1] == []:
    data.pop(len(data)-1)
matrix = np.asarray(data)

def show_point2(x,n1,n2):
    print(x[0:50,0])
    plt.plot(x[0:50,n1],x[0:50,n2],'ro',x[50:100,n1],x[50:100,n2],'b+')
    plt.show()

def show_point(matrix,n1,n2):
    for i in range(len(matrix)):
        if matrix[i][4] < 0:
            color = "red"
        else:
            color = 'blue'
        plt.scatter(matrix[i][n1], matrix[i][n2], c=color)
    plt.show()
    plt.close()

#show_point2(matrix,0,3)
def array_multiply_int(array,n):
    for i in range(len(array)):
        array[i] = array[i]*n
    return array

def perceptron(matrix, labels):
    n = len(matrix[0])
    weights = np.random.rand(1, n)
    print(weights)
    tho = 0.01
    for _ in range(400):
        delta = 0
        for i in range(len(matrix)):
            cost = labels[i]*weights[0].T.dot(matrix[i])
            if cost <= 0:
                delta += labels[i]*matrix[i]
        weights[0] = weights[0] + tho*delta
    return weights[0]


def train_perceptron(matrix,n1,n2):
    mat = []
    labels = []
    n = len(matrix[0])
    for i in range(len(matrix)):
        mat.append(np.ndarray.tolist(np.hstack(([1],matrix[i]))))
        labels.append(mat[i].pop(n))
    values = []
    for i in range(len(mat)):
        values.append([mat[i][0],mat[i][n1], mat[i][n2]])
    final = perceptron(np.asarray(values),np.asarray(labels))
    print(final)

    for i in range(len(matrix)):
        if labels[i]< 0:
            color = "red"
        else:
            color = 'blue'
        plt.scatter(matrix[i][n1-1], matrix[i][n2-1], c=color)

    z = np.linspace(0, 8, 10)
    separator_equation = (-final[1]*z-final[0])/final[2]
    plt.plot(z, separator_equation, 'b')
    plt.show()
    plt.close()
    return final

def merge_array(ar1,ar2):
    ar = []
    for x in range(len(ar1)):
        ar.append(ar1[x])
    for x in range(len(ar2)):
        ar.append(ar2[x])
    return ar
    
train = merge_array(matrix[0:25], matrix[50:75])
test = merge_array(matrix[25:50], matrix[75:100])

final = train_perceptron(train,2,3)

def test_perceptron(final,test,n1,n2):
    misclassified = 0
    values = []
    for i in range(len(test)):
        values.append([test[i][0],test[i][n1], test[i][n2]])

    for i in range(len(values)):
        cost = final.T.dot(values[i])
        if cost <= 0:
            misclassified += 1
    return misclassified / len(values) * 100

print("erreur:",test_perceptron(final,test,1,2),"%")
