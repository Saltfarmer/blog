---
title: "Introduction Neural Network"
header : 
  image : /assets/images/tensorflow.png
categories:
  - Machine Learning
  - Deep Learning
tags:
  - Neural Network
  - Tensorflow
---

You can import the library:


```python
import tensorflow as tf
```

### Simple Constants

Let's show how to create a simple constant with Tensorflow, which TF stores as a tensor object:


```python
hello = tf.constant('Hello World')
```


```python
type(hello)
```




    tensorflow.python.framework.ops.Tensor




```python
x = tf.constant(100)
```


```python
type(x)
```




    tensorflow.python.framework.ops.Tensor



### Running Sessions

Now you can create a TensorFlow Session, which is a class for running TensorFlow operations.

A `Session` object encapsulates the environment in which `Operation`
objects are executed, and `Tensor` objects are evaluated. For example:


```python
sess = tf.Session()
```


```python
sess.run(hello)
```




    b'Hello World'




```python
type(sess.run(hello))
```




    bytes




```python
sess.run(x)
```




    100




```python
type(sess.run(x))
```




    numpy.int32



## Operations

You can line up multiple Tensorflow operations in to be run during a session:


```python
x = tf.constant(2)
y = tf.constant(3)
```


```python
with tf.Session() as sess:
    print('Operations with Constants')
    print('Addition',sess.run(x+y))
    print('Subtraction',sess.run(x-y))
    print('Multiplication',sess.run(x*y))
    print('Division',sess.run(x/y))
```

    Operations with Constants
    Addition 5
    Subtraction -1
    Multiplication 6
    Division 0.666666666667


#### Placeholder

You may not always have the constants right away, and you may be waiting for a constant to appear after a cycle of operations. **tf.placeholder** is a tool for this. It inserts a placeholder for a tensor that will be always fed.

**Important**: This tensor will produce an error if evaluated. Its value must be fed using the `feed_dict` optional argument to `Session.run()`,
`Tensor.eval()`, or `Operation.run()`. For example, for a placeholder of a matrix of floating point numbers:

    x = tf.placeholder(tf.float32, shape=(1024, 1024))

Here is an example for integer placeholders:


```python
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
```


```python
x
```




    <tf.Tensor 'Placeholder_2:0' shape=<unknown> dtype=int16>




```python
type(x)
```




    tensorflow.python.framework.ops.Tensor



#### Defining Operations


```python
add = tf.add(x,y)
sub = tf.sub(x,y)
mul = tf.mul(x,y)
```

Running operations with variable input:


```python
d = {x:20,y:30}
```


```python
with tf.Session() as sess:
    print('Operations with Constants')
    print('Addition',sess.run(add,feed_dict=d))
    print('Subtraction',sess.run(sub,feed_dict=d))
    print('Multiplication',sess.run(mul,feed_dict=d))
```

    Operations with Constants
    Addition 50
    Subtraction -10
    Multiplication 600


Now let's see an example of a more complex operation, using Matrix Multiplication. First we need to create the matrices:


```python
import numpy as np
# Make sure to use floats here, int64 will cause an error.
a = np.array([[5.0,5.0]])
b = np.array([[2.0],[2.0]])
```


```python
a
```




    array([[ 5.,  5.]])




```python
a.shape
```




    (1, 2)




```python
b
```




    array([[ 2.],
           [ 2.]])




```python
b.shape
```




    (2, 1)




```python
mat1 = tf.constant(a)
```


```python
mat2 = tf.constant(b)
```

The matrix multiplication operation:


```python
matrix_multi = tf.matmul(mat1,mat2)
```

Now run the session to perform the Operation:


```python
with tf.Session() as sess:
    result = sess.run(matrix_multi)
    print(result)
```

    [[ 20.]]

That is all for now! Next we will expand these basic concepts to construct out own Multi-Layer Perceptron model!

# Tensorflow with ContribLearn

As we saw previously how to build a full Multi-Layer Perceptron model with full Sessions in Tensorflow. Unfortunately this was an extremely involved process. However developers have created ContribLearn (previously known as TKFlow or SciKit-Flow) which provides a SciKit Learn like interface for Tensorflow!

It is much easier to use, but you sacrifice some level of customization of your model. Let's go ahead and explore it!

## Get the Data

We will the iris data set.

Let's get the data:


```python
from sklearn.datasets import load_iris
```


```python
iris = load_iris()
```


```python
X = iris['data']
```


```python
y = iris['target']
```


```python
y
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])




```python
y.dtype
```




    dtype('int64')



## Train Test Split


```python
from sklearn.cross_validation import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

# Contrib.learn

Let's show you how to use the simpler contrib.learn interface!


```python
import tensorflow.contrib.learn.python.learn as learn
```

There are several high level abstraction calls to models in learn, you can explore them with Tab, but we will use DNNClassifier, which stands for Deep Neural Network:


```python
classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3)#,feature_columns=feature_columns)
classifier.fit(X_train, y_train, steps=200, batch_size=32)
```




    DNNClassifier()




```python
iris_predictions = classifier.predict(X_test)
```


```python
from sklearn.metrics import classification_report,confusion_matrix
```


```python
print(classification_report(y_test,iris_predictions))
```

                 precision    recall  f1-score   support
    
              0       1.00      1.00      1.00        13
              1       1.00      0.75      0.86        16
              2       0.80      1.00      0.89        16
    
    avg / total       0.93      0.91      0.91        45


​    