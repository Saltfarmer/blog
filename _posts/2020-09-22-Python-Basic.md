---
title: "Python Basic"
header :
  image: /assets/images/python-head.jpg
comments : true
share : true
categories:
  - Python
tags:
  - Machine Learning
  - Python
 

---

Before we dive into Pandas, Numpy and Matplotlib, let's try remind us the basic of the python first. I wont cover all Python stuff because it took too much time. In this post, i will add some interesting which might useful later. Let start with Python Operator.

# Operators

## Arithmetic Operators

Arithmetic operators are used with numeric values to perform common mathematical operations

| Operator | Name           | Example |
| :------- | :------------- | :------ |
| +        | Addition       | x + y   |
| -        | Subtraction    | x - y   |
| *        | Multiplication | x * y   |
| /        | Division       | x / y   |
| %        | Modulus        | x % y   |
| **       | Exponentiation | x ** y  |
| //       | Floor division | x // y  |

## Assignment Operators

Assignment operators are used to assign values to variables

| Operator | Example | Same As    |
| :------- | :------ | :--------- |
| =        | x = 5   | x = 5      |
| +=       | x += 3  | x = x + 3  |
| -=       | x -= 3  | x = x - 3  |
| *=       | x *= 3  | x = x * 3  |
| /=       | x /= 3  | x = x / 3  |
| %=       | x %= 3  | x = x % 3  |
| //=      | x //= 3 | x = x // 3 |
| **=      | x **= 3 | x = x ** 3 |
| &=       | x &= 3  | x = x & 3  |
| \|=      | x \|= 3 | x = x \| 3 |
| ^=       | x ^= 3  | x = x ^ 3  |
| >>=      | x >>= 3 | x = x >> 3 |
| <<=      | x <<= 3 | x = x << 3 |

## Comparison Operators

Comparison operators are used to compare two values

| Operator | Name                     | Example |
| :------- | :----------------------- | :------ |
| ==       | Equal                    | x == y  |
| !=       | Not equal                | x != y  |
| >        | Greater than             | x > y   |
| <        | Less than                | x < y   |
| >=       | Greater than or equal to | x >= y  |
| <=       | Less than or equal to    | x <= y  |

## Logical Operators

Logical operators are used to combine conditional statements

| Operator | Description                                             | Example               |
| :------- | :------------------------------------------------------ | :-------------------- |
| and      | Returns True if both statements are true                | x < 5 and x < 10      |
| or       | Returns True if one of the statements is true           | x < 5 or x < 4        |
| not      | Reverse the result, returns False if the result is true | not(x < 5 and x < 10) |

## Identity and Membeship Operators

Identity operators are used to compare the objects, not if they are equal, but if they are actually the same object, with the same memory location. It also can used as Membership operators to test if a sequence is presented in an object

| Operator | Description                                            | Example    |
| :------- | :----------------------------------------------------- | :--------- |
| is       | Returns True if both variables are the same object     | x is y     |
| is not   | Returns True if both variables are not the same object | x is not y |

# Conditional Statement

## If

An "if statement" is written by using the if keyword. Example

```python
a = 33
b = 200
if b > a:
  print("b is greater than a")
```

```
b is greater than a
```

## Elif

The elif keyword is pythons way of saying "if the previous conditions were not true, then try this condition". Example

```python
a = 33
b = 33
if b > a:
  print("b is greater than a")
elif a == b:
  print("a and b are equal")
```

```
a and b are equal
```



## Else

The else keyword catches anything which isn't caught by the preceding conditions. Example

```python
a = 200
b = 33
if b > a:
  print("b is greater than a")
elif a == b:
  print("a and b are equal")
else:
  print("a is greater than b")
```

```
a is greater than b
```



# Strings

## Array of String

Using x  = "Hello World" as example

| Name              | Description                                                  | Example  | Result |
| ----------------- | ------------------------------------------------------------ | -------- | ------ |
| Array of Strings  | Pick a character from string                                 | x[0]     | 'H'    |
| String Slicing    | Return a range of characters by using the slice syntax by Specify the start index and the end index, separated by a colon, to return a part of the string. | x[2:5]   | 'llo'  |
| Negative Indexing | Use negative indexes to start the slice from the end of the string | x[-5:-2] | 'orl'  |

## String Length

To get the length of a string, use the `len()` function. For Example

```python
a = "Hello, World!"
print(len(a))
```

```
13
```

## Check String

To check if a certain phrase or character is present in a string, we can use the keywords `in` or `not in`. Example

```python
txt = "The rain in Spain stays mainly in the plain"
x = "ain" in txt
print(x)
```

```
True
```

## String Format

We cannot combine strings and numbers. But we can combine strings and numbers by using the `format()` method!. The `format()` method takes the passed arguments, formats them, and places them in the string where the placeholders `{}` are. For Example

```python
quantity = 3
itemno = 567
price = 49.95
myorder = "I want {} pieces of item {} for {} dollars."
print(myorder.format(quantity, itemno, price))
```

```
I want 3 pieces of item 567 for 49.95 dollars.
```

## String Methods

| Method                                                       | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [capitalize()](https://www.w3schools.com/python/ref_string_capitalize.asp) | Converts the first character to upper case                   |
| [casefold()](https://www.w3schools.com/python/ref_string_casefold.asp) | Converts string into lower case                              |
| [center()](https://www.w3schools.com/python/ref_string_center.asp) | Returns a centered string                                    |
| [count()](https://www.w3schools.com/python/ref_string_count.asp) | Returns the number of times a specified value occurs in a string |
| [encode()](https://www.w3schools.com/python/ref_string_encode.asp) | Returns an encoded version of the string                     |
| [endswith()](https://www.w3schools.com/python/ref_string_endswith.asp) | Returns true if the string ends with the specified value     |
| [expandtabs()](https://www.w3schools.com/python/ref_string_expandtabs.asp) | Sets the tab size of the string                              |
| [find()](https://www.w3schools.com/python/ref_string_find.asp) | Searches the string for a specified value and returns the position of where it was found |
| [format()](https://www.w3schools.com/python/ref_string_format.asp) | Formats specified values in a string                         |
| format_map()                                                 | Formats specified values in a string                         |
| [index()](https://www.w3schools.com/python/ref_string_index.asp) | Searches the string for a specified value and returns the position of where it was found |
| [isalnum()](https://www.w3schools.com/python/ref_string_isalnum.asp) | Returns True if all characters in the string are alphanumeric |
| [isalpha()](https://www.w3schools.com/python/ref_string_isalpha.asp) | Returns True if all characters in the string are in the alphabet |
| [isdecimal()](https://www.w3schools.com/python/ref_string_isdecimal.asp) | Returns True if all characters in the string are decimals    |
| [isdigit()](https://www.w3schools.com/python/ref_string_isdigit.asp) | Returns True if all characters in the string are digits      |
| [isidentifier()](https://www.w3schools.com/python/ref_string_isidentifier.asp) | Returns True if the string is an identifier                  |
| [islower()](https://www.w3schools.com/python/ref_string_islower.asp) | Returns True if all characters in the string are lower case  |
| [isnumeric()](https://www.w3schools.com/python/ref_string_isnumeric.asp) | Returns True if all characters in the string are numeric     |
| [isprintable()](https://www.w3schools.com/python/ref_string_isprintable.asp) | Returns True if all characters in the string are printable   |
| [isspace()](https://www.w3schools.com/python/ref_string_isspace.asp) | Returns True if all characters in the string are whitespaces |
| [istitle()](https://www.w3schools.com/python/ref_string_istitle.asp) | Returns True if the string follows the rules of a title      |
| [isupper()](https://www.w3schools.com/python/ref_string_isupper.asp) | Returns True if all characters in the string are upper case  |
| [join()](https://www.w3schools.com/python/ref_string_join.asp) | Joins the elements of an iterable to the end of the string   |
| [ljust()](https://www.w3schools.com/python/ref_string_ljust.asp) | Returns a left justified version of the string               |
| [lower()](https://www.w3schools.com/python/ref_string_lower.asp) | Converts a string into lower case                            |
| [lstrip()](https://www.w3schools.com/python/ref_string_lstrip.asp) | Returns a left trim version of the string                    |
| [maketrans()](https://www.w3schools.com/python/ref_string_maketrans.asp) | Returns a translation table to be used in translations       |
| [partition()](https://www.w3schools.com/python/ref_string_partition.asp) | Returns a tuple where the string is parted into three parts  |
| [replace()](https://www.w3schools.com/python/ref_string_replace.asp) | Returns a string where a specified value is replaced with a specified value |
| [rfind()](https://www.w3schools.com/python/ref_string_rfind.asp) | Searches the string for a specified value and returns the last position of where it was found |
| [rindex()](https://www.w3schools.com/python/ref_string_rindex.asp) | Searches the string for a specified value and returns the last position of where it was found |
| [rjust()](https://www.w3schools.com/python/ref_string_rjust.asp) | Returns a right justified version of the string              |
| [rpartition()](https://www.w3schools.com/python/ref_string_rpartition.asp) | Returns a tuple where the string is parted into three parts  |
| [rsplit()](https://www.w3schools.com/python/ref_string_rsplit.asp) | Splits the string at the specified separator, and returns a list |
| [rstrip()](https://www.w3schools.com/python/ref_string_rstrip.asp) | Returns a right trim version of the string                   |
| [split()](https://www.w3schools.com/python/ref_string_split.asp) | Splits the string at the specified separator, and returns a list |
| [splitlines()](https://www.w3schools.com/python/ref_string_splitlines.asp) | Splits the string at line breaks and returns a list          |
| [startswith()](https://www.w3schools.com/python/ref_string_startswith.asp) | Returns true if the string starts with the specified value   |
| [strip()](https://www.w3schools.com/python/ref_string_strip.asp) | Returns a trimmed version of the string                      |
| [swapcase()](https://www.w3schools.com/python/ref_string_swapcase.asp) | Swaps cases, lower case becomes upper case and vice versa    |
| [title()](https://www.w3schools.com/python/ref_string_title.asp) | Converts the first character of each word to upper case      |
| [translate()](https://www.w3schools.com/python/ref_string_translate.asp) | Returns a translated string                                  |
| [upper()](https://www.w3schools.com/python/ref_string_upper.asp) | Converts a string into upper case                            |
| [zfill()](https://www.w3schools.com/python/ref_string_zfill.asp) | Fills the string with a specified number of 0 values at the beginning |

# Dictionary

A dictionary is a collection which is unordered, changeable and indexed. In Python dictionaries are written with curly brackets, and they have keys and values. Example

```python
thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
print(thisdict)
```

```
{'brand': 'Ford', 'model': 'Mustang', 'year': 1964}
```

## Dictionary Methods

| Method                                                       | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [clear()](https://www.w3schools.com/python/ref_dictionary_clear.asp) | Removes all the elements from the dictionary                 |
| [copy()](https://www.w3schools.com/python/ref_dictionary_copy.asp) | Returns a copy of the dictionary                             |
| [fromkeys()](https://www.w3schools.com/python/ref_dictionary_fromkeys.asp) | Returns a dictionary with the specified keys and value       |
| [get()](https://www.w3schools.com/python/ref_dictionary_get.asp) | Returns the value of the specified key                       |
| [items()](https://www.w3schools.com/python/ref_dictionary_items.asp) | Returns a list containing a tuple for each key value pair    |
| [keys()](https://www.w3schools.com/python/ref_dictionary_keys.asp) | Returns a list containing the dictionary's keys              |
| [pop()](https://www.w3schools.com/python/ref_dictionary_pop.asp) | Removes the element with the specified key                   |
| [popitem()](https://www.w3schools.com/python/ref_dictionary_popitem.asp) | Removes the last inserted key-value pair                     |
| [setdefault()](https://www.w3schools.com/python/ref_dictionary_setdefault.asp) | Returns the value of the specified key. If the key does not exist: insert the key, with the specified value |
| [update()](https://www.w3schools.com/python/ref_dictionary_update.asp) | Updates the dictionary with the specified key-value pairs    |
| [values()](https://www.w3schools.com/python/ref_dictionary_values.asp) | Returns a list of all the values in the dictionary           |

# Sets

A set is a collection which is unordered and unindexed. In Python, sets are written with curly brackets. Example

```python
thisset = {"apple", "banana", "cherry"}
print(thisset)
```

```
{'cherry', 'apple', 'banana'}
```

## Set Method

| Method                                                       | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [add()](https://www.w3schools.com/python/ref_set_add.asp)    | Adds an element to the set                                   |
| [clear()](https://www.w3schools.com/python/ref_set_clear.asp) | Removes all the elements from the set                        |
| [copy()](https://www.w3schools.com/python/ref_set_copy.asp)  | Returns a copy of the set                                    |
| [difference()](https://www.w3schools.com/python/ref_set_difference.asp) | Returns a set containing the difference between two or more sets |
| [difference_update()](https://www.w3schools.com/python/ref_set_difference_update.asp) | Removes the items in this set that are also included in another, specified set |
| [discard()](https://www.w3schools.com/python/ref_set_discard.asp) | Remove the specified item                                    |
| [intersection()](https://www.w3schools.com/python/ref_set_intersection.asp) | Returns a set, that is the intersection of two other sets    |
| [intersection_update()](https://www.w3schools.com/python/ref_set_intersection_update.asp) | Removes the items in this set that are not present in other, specified set(s) |
| [isdisjoint()](https://www.w3schools.com/python/ref_set_isdisjoint.asp) | Returns whether two sets have a intersection or not          |
| [issubset()](https://www.w3schools.com/python/ref_set_issubset.asp) | Returns whether another set contains this set or not         |
| [issuperset()](https://www.w3schools.com/python/ref_set_issuperset.asp) | Returns whether this set contains another set or not         |
| [pop()](https://www.w3schools.com/python/ref_set_pop.asp)    | Removes an element from the set                              |
| [remove()](https://www.w3schools.com/python/ref_set_remove.asp) | Removes the specified element                                |
| [symmetric_difference()](https://www.w3schools.com/python/ref_set_symmetric_difference.asp) | Returns a set with the symmetric differences of two sets     |
| [symmetric_difference_update()](https://www.w3schools.com/python/ref_set_symmetric_difference_update.asp) | inserts the symmetric differences from this set and another  |
| [union()](https://www.w3schools.com/python/ref_set_union.asp) | Return a set containing the union of sets                    |
| [update()](https://www.w3schools.com/python/ref_set_update.asp) | Update the set with the union of this set and others         |

# Tuple

A tuple is a collection which is ordered and **unchangeable**. In Python tuples are written with round brackets. Example

```python
thistuple = ("apple", "banana", "cherry")
print(thistuple)
```

```
('apple', 'banana', 'cherry')
```

## Tuple Methods

| Method                                                       | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [count()](https://www.w3schools.com/python/ref_tuple_count.asp) | Returns the number of times a specified value occurs in a tuple |
| [index()](https://www.w3schools.com/python/ref_tuple_index.asp) | Searches the tuple for a specified value and returns the position of where it was found |

# Iteration

## For

A for loop is used for iterating over a sequence (that is either a list, a tuple, a dictionary, a set, or a string). This is less like the for keyword in other programming languages, and works more like an iterator method as found in other object-orientated programming languages. With the for loop we can execute a set of statements, once for each item in a strings, list, tuple, and set. Example

```python
fruits = ["apple", "banana", "cherry"]
for x in fruits:
  print(x)
```

```
apple
banana
cherry
```

## Range() Function

To loop through a set of code a specified number of times, we can use the range() function/ The range() function returns a sequence of numbers, starting from 0 by default, and increments by 1 (by default), and ends at a specified number. Example

```python
for x in range(2, 30, 3):
  print(x)
```

```
2
5
8
11
14
17
20
23
26
29
```

## While

With the while loop we can execute a set of statements as long as a condition is true. Example

```python
i = 1
while i < 6:
  print(i)
  i += 1
```

```
1
2
3
4
5
```

# Function

A function is a block of code which only runs when it is called. You can pass data, known as parameters, into a function. A function can return data as a result. Example

```python
def my_function():
  print("Hello from a function")
  
my_function()
```

```
Hello from a function
```

If you using arguments in function

```python
def my_function(a, b, c):
    return a+b+c

my_function(1,2,3)
```

```
6
```

# Lambda Expression

A lambda function is a small anonymous function. A lambda function can take any number of arguments, but can only have one expression. The power of lambda is better shown when you use them as an anonymous function inside another function. Say you have a function definition that takes one argument, and that argument will be multiplied with an unknown number:

## Syntax

> lambda <u>*arguments*</u> : <u>*expression*</u>

Example

```python
x = lambda a : a + 10
print(x(5))
```

```
15
```



