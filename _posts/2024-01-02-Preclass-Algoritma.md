---
title: "Preclass Bootcamp Algorit.ma BFLP Audit"
header :
  teaser: /assets/images/Algoritma.png
comments : true
share : true
categories:
  - Data Science
tags:
  - Anaconda
  - Python

---

Well this post (I hope I can make it as a series) will be my personal notes and documentation of data science bootcamp session from **Algorit.ma**. Please notes that it will be different than usual I write it with my personal flavor and focus on my perspective. In this post of preparation and preclass Bootcamp session, I will put it simple step by step from my perspective, so it wont be the same as the instructor.

## Installing Miniconda
For my Python installation, I will use **Miniconda** instead of Anaconda. The reason I am not a big fan of Anaconda Navigator. Even if it possible, I just wanted to install the packages by `PyPI` installer instead. However, just to make sure and make it easier for me to create a new environment, I will using Miniconda. You can download it [here](https://docs.conda.io/projects/miniconda/en/latest/index.html). Keep going next like you install most programs. For **Windows** user, please make sure to check Add Anaconda to my PATH.

![](https://i.ibb.co/wzN3kLj/2024-01-02-20-01-14-Miniconda3-py310-23-10-0-1-64-bit-Setup.jpg)

## Anaconda Prompt and creating New Environment
Now after you completed installation of **Miniconda**, please check on your desktop that you can find Anaconda Prompt.

![](https://i.ibb.co/CWnqQMq/2024-01-02-20-07-21-2024-01-02-Preclass-Algoritma-md-blog-Visual-Studio-Code.jpg)

After that it is good practice to update your **Conda**. Run the command below to update your **Conda**:

```
conda update -n base -c defaults conda
```

Now, the next step is to create new Conda environment. The reason is to make sure the codes work on specific environment and Python version just in case shit happens.
Here the commands to create new environment.

```
conda create -n myenv python=<python_version>
```

In this case, you should replace `myenv` with any environment name you wanted. In my case, I use **Algoritma** as the name of the environment.  then the `<python_version>` part, I am using Python 3.8 instead because the Instructor wanted to use the minimum of Python version ">=3.8". After you create the new environment, you can run the environment by running:

```
conda activate algoritma
```

## Installing Jupyter Notebook

Then the next step is to install **Jupyter Notebook**. Jupyter Notebook is an interactive web application that allows you to create and share documents that contain both code and rich text elements, like markdown and LaTex. This makes it a popular tool for data science, machine learning, and scientific computing. 

With Jupyter Notebook, you can write code, run it, and see the results immediately. You can also add text, images, and other multimedia to your documents to make them more informative and visually appealing. This makes it a great tool for exploring data, building models, and presenting your findings.

Now now, after you activate your specific environment, you can install jupyter notebook with this command:

```
conda install -c conda-forge notebook
```

That looks different than usual because this command installs Jupyter Notebook from the **conda-forge** channel. Just in case you have a problem with `chardet`, you can manually install it with command below:

```
pip install chardet
```

And then back to the step where you install jupyter notebook. Then you can try to open your Jupyter notebook by run this command below:

```
jupyter notebook
```

Now you can see the Jupyter Notebook in your default browser. Next step is to install all prerequirement Python libraries.

## Installing prerequirement library
It is VERY SIMPLE. Just install it all LOL. Run this command and wait for it to be installed.

```
conda install pandas numpy matplotlib ipykernel seaborn openpyxl
scikit-learn plotly imbalanced-learn gower statsmodels nbformat
```

In case you dont why do you need to install all of those libraries, I dont even know. Usually I just install `pandas`, `numpy`, `matplotlib`, `scikit-learn`, and `statsmodels`. Now if you are curious, you can check it by yourself in pypi.org on each library.


