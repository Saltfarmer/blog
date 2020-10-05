---
title: "Python Crash Course Exercise"
header :
  image: /assets/images/python-head.jpg
comments : true
share : true
categories:
  - Python
tags:
  - Python
  - Exercise
 

---

This week I will dedicate my time to solve all exercise from [Jose Portilla Python for Data Science Bootcamp](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/). Today i will complete some exercise from Python Basic until Finance Project. If you want to solve it all by yourself, you can download notebooks file [here](https://drive.google.com/file/d/1nkbR1Q2XYwigiQ0n8jUMi7NJlmmvB0CH/view?usp=sharing)

</break>

</break>

</break>

</break>

</break>

</break>

</break>

</break>

</break>

</break>

</break>

</break>

</break>

</break>

Now Lets get started

## Exercises

Answer the questions or complete the tasks outlined in bold below, use the specific method described if applicable.

** What is 7 to the power of 4?**


```python
7 ** 4
```




    2401



** Split this string:**

    s = "Hi there Sam!"

**into a list. **


```python
s = "Hi there Sam!"
```


```python
s.split()
```




    ['Hi', 'there', 'Sam!']



** Given the variables:**

    planet = "Earth"
    diameter = 12742

** Use .format() to print the following string: **

    The diameter of Earth is 12742 kilometers.


```python
planet = "Earth"
diameter = 12742
```


```python
print("The diameter of {p} is {d} kilometers".format(p=planet,d=diameter))
```

    The diameter of Earth is 12742 kilometers


** Given this nested list, use indexing to grab the word "hello" **


```python
lst = [1,2,[3,4],[5,[100,200,['hello']],23,11],1,7]
```


```python
lst[3][1][2][0]
```




    'hello'



** Given this nested dictionary grab the word "hello". Be prepared, this will be annoying/tricky **


```python
d = {'k1':[1,2,3,{'tricky':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}
```


```python
d['k1'][3]['tricky'][3]['target'][3]
```




    'hello'



** What is the main difference between a tuple and a list? **


```python
"Tupple is immutable"
```




    'Tupple is immutable'



** Create a function that grabs the email website domain from a string in the form: **

    user@domain.com

**So for example, passing "user@domain.com" would return: domain.com**


```python
def getDomain(s):
    return s.split('@')[1]
```


```python
getDomain("user@domain.com")
```




    'domain.com'



** Create a basic function that returns True if the word 'dog' is contained in the input string. Don't worry about edge cases like a punctuation being attached to the word dog, but do account for capitalization. **


```python
def findDog(s):
    return "dog" in s.lower()
```


```python
findDog("Is there any Dog there ?")
```




    True



** Create a function that counts the number of times the word "dog" occurs in a string. Again ignore edge cases. **


```python
def countDog(s):
    count = 0
    for word in s.lower().split():
        if "dog" in word:
            count += 1
    return count
```


```python
countDog("Those doggies is dodging me")
```




    1



** Use lambda expressions and the filter() function to filter out words from a list that don't start with the letter 's'. For example:**

    seq = ['soup','dog','salad','cat','great']

**should be filtered down to:**

    ['soup','salad']


```python
seq = ['soup','dog','salad','cat','great']
```


```python
list(filter(lambda s: s[0]=='s',seq))
```




    ['soup', 'salad']



### Final Problem

**You are driving a little too fast, and a police officer stops you. Write a function
  to return one of 3 possible results: "No ticket", "Small ticket", or "Big Ticket". 
  If your speed is 60 or less, the result is "No Ticket". If speed is between 61 
  and 80 inclusive, the result is "Small Ticket". If speed is 81 or more, the result is "Big    Ticket". Unless it is your birthday (encoded as a boolean value in the parameters of the function) -- on your birthday, your speed can be 5 higher in all 
  cases. **


```python
def ticket(speed, birthday):
    if birthday is True:
        speed -= 5
    if speed > 80:
        return "Big Ticket"
    elif speed >60:
        return "Small Ticket"
    else :
        return "No Ticket"
```


```python
ticket(66, True)
```




    'Small Ticket'




```python
ticket(81, False)
```




    'Big Ticket'



# Great job!