  ---
title: "BBC Recommendation System (Content-Based)"
header : https://ichef.bbci.co.uk/images/ic/1920x1080/p09xtmrp.jpg
comments : true
share : true
categories:
  - Python
  - Recommendation-System
  - Text-mining
tags:
  - Python
---



## Introduction

Recommender systems aim to predict users' interests and recommend stuff that is interesting for the user. Data are required for recommender systems from either the user (collaborative filtering), service provider (content-based filtering), or both (hybrid filtering). In this project, I am trying to create a prototype of a recommendation system based on the articles from BBC.

The British Broadcasting Corporation (BBC) is the national broadcaster of the United Kingdom. Headquartered at Broadcasting House in London, it is the world's oldest national broadcaster, and the largest broadcaster in the world by the number of employees, employing over 22,000 staff in total, of whom approximately 19,000 are in public-sector broadcasting. They produce programs and services for audiences throughout the UK. They also produce content that can be enjoyed across the globe.

There are many stakeholders from BBC starting including their service provider, viewer/user/reader, Board of Executives, Commercial Provider, and a lot of stakeholders that need to be mentioned. In this project, the main stakeholders are The service provider and the viewer. The service provider in this case is the one who provides BBC with the content of the news or informative videos. On the other hand, the viewer is the one who enjoys and watches the content that has been provided by BBC.

In this recommender system prototype, I am trying to apply transparency and autonomy for the user to choose or manipulate the recommendation they could get. The recommender will be provided only by the content from BBC metadata that has been scrapped before. Then the system starts recommending random recommendations at first and then the next recommendation is adjusted by the user.

## Literature Reviews

The content-based news recommendation system is already has been researched by [Kompan & Bielikova, 2010]([Content-Based News Recommendation | SpringerLink](https://link.springer.com/chapter/10.1007/978-3-642-15208-5_6)). The paper recommendation system uses Title, Article content, Names & Places, Keywords, Category, and Coleman-Liau Index (CLI). For the recommendation part, they use Cosine-Similarity as similarity measurement as a recommendation of one article to another article. In my case, **I am using K-Means clustering as a recommendation for the related content**.

[Kim & Ahn, 2008](https://www.sciencedirect.com/science/article/pii/S0957417406004076) did a recommender system using K-Means. the result is that **K-means clustering may improve segmentation performance in comparison to other typical clustering algorithms**. In addition, their study validated the usefulness of the proposed model as a preprocessing tool for recommendation systems.

From the perspective of BBC as a provider, based on the [2020-2021 Annual Report](https://downloads.bbc.co.uk/aboutthebbc/reports/annualreport/2020-21.pdf#page=20) I can get the value from the provider. According to the report, there are 5 things to measure audience performance started from. One of them is to **provide impartial news and information to help people to understand and engage with the world around them**. This recommendation system will provide you with more engaging news and interesting news depending either on only the same cluster or the same cluster and the same category.

According to [Friedman, 1996](https://dl.acm.org/doi/pdf/10.1145/242485.242493) minimizing bias when considering **user autonomy in a design also likely leads to a larger market share** because such systems are typically accessible to a greater diversity of users. This also helps and is related to other values that the service provider likes to get.

Using the Value Sensitive Design (VSD) method as a point of departure, [Jacob, et al, 2020](https://link.springer.com/chapter/10.1007/978-3-030-50334-5_1) explores how VSD can be used in the context of transparency. More precisely, it is investigated if the VSD Envisioning Cards facilitate transparency as a pro-ethical condition. Therefore, it is proposed that a transparency card be added to the Envisioning Card deck. **It is concluded that a lightweight version of VSD seems useful in engaging larger audiences**. This means the transparency value also helps the value from BBC's perspective as a stakeholder.

From the survey of [Public Opinion on the BBC and BBC News](https://www.ofcom.org.uk/__data/assets/pdf_file/0014/58001/bbc-annex2.pdf), **trustworthiness is one of the most important values that influence the user to choose a news provider**. With an explanation of the system, users can see the transparency of how the users can choose their recommendations. This will create a sense of trustworthiness towards the recommendation system.

## Method

In the very first step I choose the value that I need to consider when creating the prototype of the Recommendation System. Here there are two main stakeholders that I consider when creating the prototype. The service provider is BBC and the user. Then I look for the value from those two. From the service provider perspective, they value providing creative, highest quality, and distinctive output and services to users. From the user perspective, they consider trustworthiness and transparency.

The next step is to gather all of the metadata that might be useful for my recommendation system. Starting from the Title, Description, Images, Url, Category, and Keywords. I am using the `BeautifulSoup4` library to get all metadata from all the articles that have been gathered or provided beforehand. Then I save all the metadata from all articles to process it later as a recommendation system later as a Comma Separable Value file.

The final step is I put the design of my recommendation system that has in mind into the code using `Streamlit`. In the system, first I provide the user with a completely random recommendation as a starter. Then using K-Means clustering, I am trying to provide the user the recommendation from BBC content according to cluster and the genre of the article. K-Means clustering creates a `k` cluster value from vectorized value from the description of the content.

The clustering itself works by the `k` cluster the user is free to choose to start from 2-10 clusters. Then, the system will vectorize the description of the article with the TF-IDF vectorization of all articles. The vectorized words are important for the program to understand the value of words in every article. The values then can be used for the K-Means algorithm to create a cluster.

## Interface Design

First, the system will show the user the explanation of the source of the dataset and how I collect that dataset. Then it will show you some samples of the dataset. It will help at supporting of transparency value of the user as a stakeholder.

The next step of the recommendation system is to show complete a random recommendation of content from the BBC. The interface will show the user the poster, title, small description, keyword of the content, and an option to watch the content if the user is interested in the recommendation. This interface will work as the starting point for the other recommendations.

The second part is where when the user gives their transparency and autonomy. The transparency part is when the user already explained how is the recommendation system going to work. And then the user could choose a specific value and autonomy on how many clusters they choose on K-means clustering. Later on, it will show another sample of clustered data.

In this part, the recommendation system will give you 5 random recommendations based on the same cluster as content from the starting point. In this design, users can click the recommendation so they make the content into the starting point and create a new recommendation.

Finally, the user also can get a different type of recommendation by looking at the same category of content. It will give the user more recommendations and more autonomy from the user.

## Conclusion

So in the summary, the system is very limited. Both from a data perspective and a value-sensitive design perspective.

First, the metadata doesn't have a lot of useful features that might be important for the model. The metadata that has been provided lacks some important features that might be helpful to create a more precise recommendation. There might be also redundancy in the features from the cluster of description and category.

Second, the lack of data from the user makes the consideration of the value of the user as a stakeholder has been ignored. Because the generated metadata has been scrapped from only what users can see on the service provider, it might be ignoring what the service provider can see on their user.

Third, there is no application of value from service provider stakeholders. The information itself is not publicly open without doing a survey on the service provider's opinion regarding of what is their actual value.

## Reference

1. Kompan, M., & Bieliková, M. (2010, September). Content-based news recommendation. In *International conference on electronic commerce and web technologies* (pp. 61-72). Springer, Berlin, Heidelberg.

2. Kim, K. J., & Ahn, H. (2008). A recommender system using GA K-means clustering in an online shopping market. *Expert systems with applications*, *34*(2), 1200-1209.

3. *BBC group annual report and accounts 2020/21*. (2021). Https://Www.Bbc.Com/. Retrieved March 18, 2022, from https://downloads.bbc.co.uk/aboutthebbc/reports/annualreport/2020-21.pdf

4. Friedman, B. (1996). Value-sensitive design. *interactions*, *3*(6), 16-23.

5. Dexe, J., Franke, U., Nöu, A. A., & Rad, A. (2020, July). Towards increased transparency with value sensitive design. In *International Conference on Human-Computer Interaction* (pp. 3-15). Springer, Cham.

6. *Public opinion on the BBC and BBC news*. (2011b, November). Https://Www.Ofcom.Org.Uk/. Retrieved March 18, 2022, from https://www.ofcom.org.uk/__data/assets/pdf_file/0014/58001/bbc-annex2.pdf
  
