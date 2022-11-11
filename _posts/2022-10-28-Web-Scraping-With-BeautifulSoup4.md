---
title: "Web Scraping with BeautifulSoup4"
header: 
  teaser : /assets/images/BS4.png
categories:
  - Python
  - Data Engineering
tags:
  - Python
  - BeautifulSoup4
---

The surge of available data we can find on the internet is insane. With this surge, data analytics has become a hugely important part of the way organizations are run. And while data has many sources, its biggest repository is on the internet. As the fields of big data analytics, artificial intelligence and machine learning grow, everyone needs data analysts who can scrape the web in the most eloquent ways.

## What is web scraping?
Web scraping, web harvesting, or web data extraction is data scraping used for extracting data from websites. Web scraping programs may directly access the World Wide Web using the Hypertext Transfer Protocol or a web browser. While web scraping can be done manually by a software user, the term typically refers to automated processes implemented using a bot or a web crawler. It is a form of copying in which specific data is gathered and copied from the web, typically into a central local database or spreadsheet, for later retrieval or analysis. The types of data you can scrape on the internet could be any. Everything that computer programs can read on the internet, also can be scraped. Websites that don't wish to be crawled or found by search engines can use tools like the `robots.txt` file to request bots not index a website or only index portions of it.

## How does it work?
Every program has different methods. But most of them follow these 3 main principles :
1. Making HTTP Request to the website that you want to scrap
2. Extracting (or parsing) the code of the website
3. Find the specific data that you want to scrap this code

## What tools do you use to scrape the data?
In this post, I want to focus on scraping data using the library from Python. The 3 main library that you can use to scrape data is `Pandas` for data manipulation, `Request` to get access to the websites, and `BeautifulSoup4` to parse the code of the website for scraping data. There might be different alternatives library that you can use to scrape data such as Selenium Web Driver and Scrapy.

## What is BeautifulSoup4 ?
Beautiful Soup is a Python library for pulling data out of HTML and XML files. It works with your favorite parser to provide idiomatic ways of navigating, searching, and modifying the parse tree. It commonly saves programmers hours or days of work. Beautiful Soup is a Python library designed for quick turnaround projects like screen-scraping. Three features make it powerful:

- Beautiful Soup provides a few simple methods and Pythonic idioms for navigating, searching, and modifying a parse tree: a toolkit for dissecting a document and extracting what you need. It doesn't take much code to write an application
- Beautiful Soup automatically converts incoming documents to Unicode and outgoing documents to UTF-8. You don't have to think about encodings unless the document doesn't specify an encoding and Beautiful Soup can't detect one. Then you just have to specify the original encoding.
- Beautiful Soup sits on top of popular Python parsers like lxml and html5lib, allowing you to try out different parsing strategies or trade speed for flexibility.

Beautiful Soup parses anything you give it and does the tree traversal stuff for you. You can tell it "Find all the links", "Find all the links of class external link", "Find all the links whose URLs match "foo.com", or "Find the table heading that's got bold text, then give me that text." Valuable data that was once locked up in poorly-designed websites is now within your reach. Projects that would have taken hours to take only minutes with Beautiful Soup.

## Quick way to start web scraping with BS4
In this quick tutorial, I will make use of [Toscape](https://toscrape.com/), a web scraping sandbox, ideal for both beginners and advanced scrapers. It’s one of the most popular websites to try out web scraping tools. The website is divided into two parts. We will focus on the first part; a fictional bookstore that offers thousands of books to scrape. [Books.toscrape.com](https://Books.toscrape.com) allows you to practice many basic skills like extracting data – title, stock availability, price, and authors. It only includes static content, so you can use simple libraries like `Requests` and `Beautiful Soup`.

To start with, we need to retrieve the website code using `requests` library in Python. When you send a **GET** request to a website, it returns a file with two things: a response code and the response data. To create a **GET** request, we first import the Python requests library. Making a **GET** request with this library requires one thing: the URL of the website. Try to request something from a non-existing URL and inspect the status code. The response object you have created has multiple properties, of which a new overview is shown at: https://www.w3schools.com/python/ref_requests_response.asp

```python
# Import requests
import requests
# Write some code here to request the bookstore website
response = requests.get('https://books.toscrape.com/')
# Inspect the text of the response
if response.status_code == 200:
    print('Success!')
elif response.status_code == 404:
    print('Not Found.')
```

The response to the `get` request is one big HTML file. In order to extract data from the HTML file, we need to parse the HTML document into a tree of Python objects. We will use BeautifulSoup as a parser. A nice image to illustrate the use of BeautifulSoup is below.

![Web Scraping](https://mydatacareer.com/wp-content/uploads/2017/05/Web-scraping1-1-scaled-1.jpg)

```python
# Import BeautifulSoup from bs4 
from bs4 import BeautifulSoup
# Parse the response to a soup object 
soup = BeautifulSoup(response.content)
print(soup.find_all('article', class_='product_pod'))
```

```
Output exceeds the size limit. Open the full output data in a text editor
[<article class="product_pod">
<div class="image_container">
<a href="frankenstein_20/index.html"><img alt="Frankenstein" class="thumbnail" src="../media/cache/00/25/0025515e987a1ebd648773f9ac70bfe6.jpg"/></a>
</div>
<p class="star-rating Two">
<i class="icon-star"></i>
<i class="icon-star"></i>
<i class="icon-star"></i>
<i class="icon-star"></i>
<i class="icon-star"></i>
</p>
<h3><a href="frankenstein_20/index.html" title="Frankenstein">Frankenstein</a></h3>
<div class="product_price">
<p class="price_color">£38.00</p>
<p class="instock availability">
<i class="icon-ok"></i>
    
        In stock
    
</p>
<form>
<button class="btn btn-primary btn-block" data-loading-text="Adding..." type="submit">Add to basket</button>
</form>
</div>
</article>, <article class="product_pod">
...
<button class="btn btn-primary btn-block" data-loading-text="Adding..." type="submit">Add to basket</button>
</form>
</div>
</article>]
```

The `find()` and `find_all()` methods are among the most powerful weapons in your arsenal. `soup.find()` is great for cases where you know there is only one element you're looking for, such as the body tag. `soup.find_all()` is the most common method you will be using in your web scraping adventures. Using this you can iterate through all of the books on the page and print their names:

```python
# Loop over the books and print to title 
for link in soup.find_all('article', class_='product_pod'):
    print(link.find('h3').find('a').get('title'))
```

```
Frankenstein
Forever Rockers (The Rocker #12)
Fighting Fate (Fighting #6)
Emma
Eat, Pray, Love
Deep Under (Walker Security #1)
Choosing Our Religion: The Spiritual Lives of America's Nones
Charlie and the Chocolate Factory (Charlie Bucket #1)
Charity's Cross (Charles Towne Belles #4)
Bright Lines
Bridget Jones's Diary (Bridget Jones #1)
Bounty (Colorado Mountain #7)
Blood Defense (Samantha Brinkman #1)
Bleach, Vol. 1: Strawberry and the Soul Reapers (Bleach #1)
Beyond Good and Evil
Alice in Wonderland (Alice's Adventures in Wonderland #1)
Ajin: Demi-Human, Volume 1 (Ajin: Demi-Human #1)
A Spy's Devotion (The Regency Spies of London #1)
1st to Die (Women's Murder Club #1)
1,000 Places to See Before You Die
```

Then we are going to store the data with `Pandas`. 

```python
# Import pandas
import pandas as pd
# Create an empty list called books 
books = []
```

The soups contain a lot of books. We use a for each loop to retrieve information from each book. More information on the ForEach loop [here](https://www.w3schools.com/python/python_for_loops.asp). An example of a for each loop that stores data to a list:

```python
# Loop over each book and append to the empty list
for link in soup.find_all('h3'):
    books.append(link.find('a').get('title')) 
books
```

```
['A Light in the Attic',
 'Tipping the Velvet',
 'Soumission',
 'Sharp Objects',
 'Sapiens: A Brief History of Humankind',
 'The Requiem Red',
 'The Dirty Little Secrets of Getting Your Dream Job',
 'The Coming Woman: A Novel Based on the Life of the Infamous Feminist, Victoria Woodhull',
 'The Boys in the Boat: Nine Americans and Their Epic Quest for Gold at the 1936 Berlin Olympics',
 'The Black Maria',
 'Starving Hearts (Triangular Trade Trilogy, #1)',
 "Shakespeare's Sonnets",
 'Set Me Free',
 "Scott Pilgrim's Precious Little Life (Scott Pilgrim #1)",
 'Rip it Up and Start Again',
 'Our Band Could Be Your Life: Scenes from the American Indie Underground, 1981-1991',
 'Olio',
 'Mesaerion: The Best Science Fiction Stories 1800-1849',
 'Libertarianism for Beginners',
 "It's Only the Himalayas"]
```

Create a data frame based on the list. The code is 

```python
# empty list
title = []
price = []
stock = []
refference = []

# Scrape all the pages
for i in range(1,51):
    response = requests.get('https://books.toscrape.com/catalogue/page-{}.html'.format(i))
    print(response)
    soup = BeautifulSoup(response.content)
    for link in soup.find_all('article', class_='product_pod'):
        title.append(link.find('h3').find('a').get('title'))
        price.append(link.find('p', class_='price_color').get_text())
        stock.append(link.find('p', class_='instock availability').get_text().strip())
        refference.append(link.find('h3').find('a').get('href'))

# remove the currency
price = [i.split('£')[1] for i in price]

df = pd.DataFrame(list(zip(title, price, stock, refference)), columns=['title', 'price', 'stock', 'refference'])
df
```

```
	title	price	stock	refference
0	A Light in the Attic	51.77	In stock	a-light-in-the-attic_1000/index.html
1	Tipping the Velvet	53.74	In stock	tipping-the-velvet_999/index.html
2	Soumission	50.10	In stock	soumission_998/index.html
3	Sharp Objects	47.82	In stock	sharp-objects_997/index.html
4	Sapiens: A Brief History of Humankind	54.23	In stock	sapiens-a-brief-history-of-humankind_996/index...
...	...	...	...	...
995	Alice in Wonderland (Alice's Adventures in Won...	55.53	In stock	alice-in-wonderland-alices-adventures-in-wonde...
996	Ajin: Demi-Human, Volume 1 (Ajin: Demi-Human #1)	57.06	In stock	ajin-demi-human-volume-1-ajin-demi-human-1_4/i...
997	A Spy's Devotion (The Regency Spies of London #1)	16.97	In stock	a-spys-devotion-the-regency-spies-of-london-1_...
998	1st to Die (Women's Murder Club #1)	53.98	In stock	1st-to-die-womens-murder-club-1_2/index.html
999	1,000 Places to See Before You Die	26.08	In stock	1000-places-to-see-before-you-die_1/index.html
1000 rows × 4 columns
```

```python
df['price'] = df['price'].astype('float')
```

Try to explore the pandas dataframe. What is the most expensive book on the webpage? Share your founding in the new features of my blog below.

  
