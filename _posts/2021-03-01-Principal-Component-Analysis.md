---
title: "Principal Component Analysis"
header : 
  teaser : /assets/images/pca.jpeg
categories:
  - Machine Learning
tags:
  - Unsupervised Learning
  - Principal Component Analysis
---

Hello guys, it's been 3 months since my last post in Machine Learning. I'll admit that I am a little bit rusty nowadays. Because of my interviews in some company last week, I am trying to dig up again a lot of things about Principal Component Analysis (PCA). So what is PCA? PCA is the process of computing the principal components (Which is mostly Data Features) and using them to perform a change in the basis of the data (or you can say the dimension of the data). 

Reducing the number of variables of a data set naturally comes at the expense of accuracy, but the trick in dimensionality reduction is to trade a little accuracy for simplicity. Because smaller data sets are easier to explore and visualize and make analyzing data much easier and faster for machine learning algorithms without extraneous variables to process.

Mainly, there are 2 ways to reduce the dimension of data. First, by **Feature Elimination**.  Feature elimination is literally what it names. We reduce the feature space by eliminating features. The good point of using this method is its simplicity and still maintain the interpretability of your variables. On the other hand, you will miss the information from the variables you've dropped.

The second way to reduce the dimension of data is by **Feature Extraction**.  Let's say that in Feature Extraction we create "new dimension" independent data where the new independent data is a combination of each "old dimension" data. PCA is a technique for Feature Extraction. The method has benefits when the [assumptions of a linear model](http://people.duke.edu/~rnau/testing.htm) require our independent variables to be independent of one another. 

So to summarize, the idea of PCA is to reduce the number of variables while preserving useful data as much as possible. 

## When you use PCA?

1. Do you want to reduce the number of variables, but aren’t able to identify variables to completely remove from consideration?
2. Do you want to ensure your variables are independent of one another?
3. Are you comfortable making your independent variables less interpretable?

If you answered “yes” to all three questions, then PCA is a good method to use. If you answered “no” to question 3, you **should not** use PCA.

## How does it work?

Eigenvalue Decomposition and Singular Value Decomposition (SVD) from linear algebra are the two main procedures used in PCA to reduce dimensionality. 

**Matrix Decomposition** is a process in which a matrix is reduced to its constituent parts to simplify a range of more complex operations. 

**Eigenvalue Decomposition** is the most used matrix decomposition method which involves decomposing a square matrix(**n\*n**) into a set of eigenvectors and eigenvalues.

**Eigenvectors** are unit vectors, which means that their length or magnitude is equal to 1.0. They are often referred to as right vectors, which simply means a column vector (as opposed to a row vector or a left vector).

**Eigenvalues** are coefficients applied to eigenvectors that give the vectors their length or magnitude. For example, a negative eigenvalue may reverse the direction of the eigenvector as part of scaling it.

Mathematically, A vector is an eigenvector of a matrix any n*n square matrix **A** if it satisfies the following equation:

$A . v =𝞴 . v$

This is called the eigenvalue equation, where **A** is an n*n parent square matrix that we are decomposing, **v** is the eigenvector of the matrix, and 𝞴 represents the eigenvalue scalar. In simpler words, the linear transformation of a vector **v** by **A** has the same effect of scaling the vector by factor 𝞴.

![](https://i.ibb.co/N3fK9rZ/nagesh-pca-6.png)



Multiplying these constituent matrices together, or combining the transformations represented by the matrices will result in the original matrix.

A decomposition operation does not result in a compression of the matrix; instead, it breaks it down into constituent parts to make certain operations on the matrix easier to perform. Like other matrix decomposition methods, 

**Eigen decomposition** is used as an element to simplify the calculation of other more complex matrix operations. Singular value decomposition is a method of decomposing a matrix into three other matrices.

![](https://i.ibb.co/2vPjZvM/nagesh-pca-7.png)

$A= USV^T$

Where:

- *A* is an *m* × *n* matrix
- *U* is an *m* × *n* *orthogonal* matrix
- *S* is an *n* × *n* *diagonal matrix*
- *V* is an *n* × *n* orthogonal matrix

## **How does dimension reduction fit into these mathematical equations?**

Well, once you have calculated eigenvalues and eigenvectors choose the important eigenvectors to form a set of principal axes.

### Selection of EigenVectors

The importance of an eigenvector is measured by the percentage of total variance explained by the corresponding eigenvalue. Suppose V1 & V2 are two eigenvectors with 40% & 10% of total variance along with their directions respectively. If asked to pick one from these two eigenvectors, our choice would be V1 because it gives us more information about data.

All eigenvectors are arranged according to their eigenvalues in descending order. Now, we have to decide how many eigenvectors to retain and for that we need to discuss two methods **Total variance explained** and **Scree Plot** for that.

### Total Variance Explained

**Total Explained variance** is used to measure the discrepancy between a model and actual data. It is the part of the model’s total variance that is explained by factors that are present.

Suppose, we have a vector of n eigenvalues(e0,..., en) sorted in descending order. Take the cumulative sum of eigenvalues at every index until the sum is greater than 95% of the total variance. Reject all eigenvalues and eigenvectors after that index.

### Scree Plot

From the scree plot, we can read off the percentage of the variance in the data explained as we add principal components. It shows the eigenvalues on the y-axis and the number of factors on the x-axis. It always displays a downward curve. The point where the slope of the curve is leveling off (the “elbow) indicates the number of factors.

![](https://i.ibb.co/pdNd9NP/nagesh-pca-8.png)