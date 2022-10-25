---
title: "Starting Material"
header :
  image: /assets/images/python-head.jpg
comments : true
share : true
categories:
  - Machine Learning
tags:
  - Machine Learning
  - IDE
 

---

Ok, now after 1 day break and dilly dally learning theory about Machine Learning and Evaluation Metric (I'm kind of regret tell the theory first because it took a lot amount of research), we will start practicing together by little by little. 

## R or Python ?

Oh, and I almost forgot you should install your Python first. If you R user, you should skip this blog and go find another blogger who focus on R. I'm not trying to start Python vs R war. The reason I'm using python is simply it is more popular and more flexible. Python includes great libraries to manipulate matrix or to code the algorithms. As a beginner, it might be easier to learn how to build a model from scratch and then switch to the functions from the machine learning libraries. You can find ton of resources to learn Python. I used to start learning Machine Learning with R in RStudio and RapidMiner, but I feel like that Python syntax is more enjoyable. The ease of using Python libraries means that we don’t have to write or maintain as much code, allowing us to focus on getting improvements on different area.

So in the end everyone have their personal preference. To start with, install python on your operating system. You can find it [here](python.org). 

## Python 2.x or Python 3.x ?

Another debate. I am using Python 3.x because it's updated and have better performance. Some of the changes made in Python 3 have actually made it easier for beginners to understand, so it's the best way to learn Python for the first time. Keep in mind that [Python 2.7 will no longer be supported](https://pythonclock.org/) after 2020, so dedicating effort to learning it at this point won't make sense.

For Windows user like me, just install your Python like normal human being by double click the Python installer. I suggest you install the stable release in case there is still bug in newer version. Then using Python Installer Packager or PIP in Command Prompt, install Jupyter. If you dont like the hassle, just simply install Anaconda in [here](https://www.anaconda.com/products/individual).

## Why i'm not Anaconda ?

It simply just too heavy for my cheap laptop. Anyway i can do same stuff in Anaconda manually with command prompt so yeah just simply too much stuff for me. After installing Python, I recommend you to install 

- Jupyter (as IDE)
- IPython (as Console)
- Pandas (as Data wrangler)
- Numpy (some statistic and matrices stuff here)
- Scipy (More advanced stuff than numpy)
- Matplotlib (Basic visualization)
- Scikit-Learn (All in one Machine Learning kit)

with PIP in command prompt or Conda (if you are using Anaconda). Then you can start your Jupyter Notebook from your browser with command 

```
jupyter notebook
```

But personally, i'm using Jupyter Lab because it is pretty much the same as Jupyter Notebook but with more convenient features. Another option is using IDE like Spyder or Rodeo for non-connection version and Google Colab for Online notebook-like version.

## Create a shortcut in desktop without opening cmd

1. Right click create new shortcut on your desktop
2. Insert "jupyter lab" or "jupyter notebook"
3. (optional) if you save your files outside C: drive, you should add "%comspec% /k d:"
4. (optional) download jupyter logo and place it on your shortcut

Then after you finished preparation, in the next few weeks lets try mastering Pandas together with me with some example to get a good grasp for it. See you tomorrow.

