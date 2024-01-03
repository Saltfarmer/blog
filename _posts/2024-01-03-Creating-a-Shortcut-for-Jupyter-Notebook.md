---
title: "Creating a shortcut for Jupyter Notebook"
header :
  teaser: /assets/images/JupyterBanner.jpg
comments : true
share : true
categories:
  - Data Science
tags:
  - Anaconda
  - Jupyter Notebook

---

This is a quick post on how to create a shortcut for Jupyter Notebook. In this case, you need to connect your PATH of your Python Conda. Here's how:

1. Open your "Edit the system environment variables"
2. Then click on "Environment Variables...:
3. Click twice on "Path"
4. Click new
5. Add `C:\Users\{YourPCUsername}\miniconda3\Scripts`
6. Donezo

## Create a shortcut

![](https://i.ibb.co/hFjQ21Q/2024-01-03-19-35-25-Whats-App.jpg)

## Add a shortcut location

Type in the following: `cmd /k conda activate 'envName' & jupyter notebook`

![](https://i.ibb.co/tmj91s3/2024-01-03-19-39-36-Create-Shortcut.png)

## Write the name of shortcut

## (Optional) add the icon

You can get the icon of jupyter notebook [here](https://icon-icons.com/icon/jupyter-app/161280) and then choose `ICO` part


