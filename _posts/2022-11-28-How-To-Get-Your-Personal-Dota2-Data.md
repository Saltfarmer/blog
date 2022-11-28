---
title: "How to get your personal Dota2 Data"
header :
  teaser: /assets/images/opendota.jpg
categories:
  - Dota
tags:
  - Random
  - Dota
  - Python
---

This is going to be a short post. This is really interesting for me personally. As a Data Scientist and avid Dota 2 player, what could be better than doing data analysis on Dota 2 matches? In this post, I used the API from [opendota.com](https://www.opendota.com/api-keys). This API is free to use at least for your personal Dota 2 data which I assume is not that much and not exceeding the free tier limits. For the data cleaning and data collection, I will use `Pandas` and `requests`.

## Get the necessary library
```python
import pandas as pd
import numpy as np
import requests
```

Check your call status just to make sure.

```python
r = requests.get('https://api.opendota.com/api')
r.status_code
```

If it is showing `200` so it is successfully accessing the API. Now put in your personal Dota2 ID. You can find it based on your profile in opendota or the ID from your in-game.

## Make a call on Dota 2 player API

```python
myDota2ID = '296360583'

r = requests.get('https://api.opendota.com/api/players/{}/matches'.format(myDota2ID))

jsondata = pd.json_normalize(r.json())
jsondata.sample(5)
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>player_slot</th>
      <th>radiant_win</th>
      <th>duration</th>
      <th>game_mode</th>
      <th>lobby_type</th>
      <th>hero_id</th>
      <th>start_time</th>
      <th>version</th>
      <th>kills</th>
      <th>deaths</th>
      <th>assists</th>
      <th>skill</th>
      <th>average_rank</th>
      <th>leaver_status</th>
      <th>party_size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>157</th>
      <td>6736613083</td>
      <td>1</td>
      <td>True</td>
      <td>1967</td>
      <td>22</td>
      <td>0</td>
      <td>119</td>
      <td>2022-09-02 10:17:16</td>
      <td>NaN</td>
      <td>5</td>
      <td>8</td>
      <td>23</td>
      <td>NaN</td>
      <td>62.0</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>4751540797</td>
      <td>130</td>
      <td>True</td>
      <td>533</td>
      <td>22</td>
      <td>0</td>
      <td>119</td>
      <td>2019-05-14 17:03:25</td>
      <td>21.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>3</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>284</th>
      <td>6646342814</td>
      <td>132</td>
      <td>False</td>
      <td>2635</td>
      <td>22</td>
      <td>0</td>
      <td>96</td>
      <td>2022-07-03 18:00:18</td>
      <td>21.0</td>
      <td>5</td>
      <td>6</td>
      <td>24</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1569</th>
      <td>5296101869</td>
      <td>1</td>
      <td>True</td>
      <td>2232</td>
      <td>22</td>
      <td>0</td>
      <td>128</td>
      <td>2020-03-16 15:02:32</td>
      <td>21.0</td>
      <td>6</td>
      <td>5</td>
      <td>21</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2504</th>
      <td>3889956318</td>
      <td>132</td>
      <td>False</td>
      <td>2760</td>
      <td>22</td>
      <td>0</td>
      <td>68</td>
      <td>2018-05-14 14:57:06</td>
      <td>21.0</td>
      <td>5</td>
      <td>9</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>

```python
jsondata.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3081 entries, 0 to 3080
Data columns (total 16 columns):
 #   Column         Non-Null Count  Dtype         
---  ------         --------------  -----         
 0   match_id       3081 non-null   int64         
 1   player_slot    3081 non-null   int64         
 2   radiant_win    3081 non-null   bool          
 3   duration       3081 non-null   int64         
 4   game_mode      3081 non-null   int64         
 5   lobby_type     3081 non-null   int64         
 6   hero_id        3081 non-null   int64         
 7   start_time     3081 non-null   datetime64[ns]
 8   version        2619 non-null   float64       
 9   kills          3081 non-null   int64         
 10  deaths         3081 non-null   int64         
 11  assists        3081 non-null   int64         
 12  skill          1483 non-null   float64       
 13  average_rank   282 non-null    float64       
 14  leaver_status  3081 non-null   int64         
 15  party_size     2543 non-null   float64       
dtypes: bool(1), datetime64[ns](1), float64(4), int64(10)
memory usage: 364.2 KB
```

So there you go the preview we gathered on my personal Dota2 matches. Of course, you could gather more data by accessing more match API based on my Dota2 personal data. It could take you a lot of time because the match details are really detailed including the different 10 players in each game and each player has their own stats. 

You could try to access every match but beware it is going to be a lot of time.

## Get the match details on every match ID based on personal data

```python
matchlist = []
for match in jsondata['match_id']:
    r = requests.get('https://api.opendota.com/api/matches/{}'.format(match))
    matchlist.append(r.json())

pd.json_normalize(matchlist[0]).columns
```

```
Index(['match_id', 'barracks_status_dire', 'barracks_status_radiant', 'chat',
       'cluster', 'cosmetics', 'dire_score', 'dire_team_id', 'draft_timings',
       'duration', 'engine', 'first_blood_time', 'game_mode', 'human_players',
       'leagueid', 'lobby_type', 'match_seq_num', 'negative_votes',
       'objectives', 'picks_bans', 'positive_votes', 'radiant_gold_adv',
       'radiant_score', 'radiant_team_id', 'radiant_win', 'radiant_xp_adv',
       'skill', 'start_time', 'teamfights', 'tower_status_dire',
       'tower_status_radiant', 'version', 'replay_salt', 'series_id',
       'series_type', 'players', 'patch', 'region', 'replay_url'],
      dtype='object')
```
Creating the Dataframe and normalizing the JSON data from `matchlist` to save it later into `.csv`.

```python
matches_df = pd.DataFrame()

for match in matchlist:
    matches_df = pd.concat([matches_df, pd.json_normalize(match)], axis=0) 

matches_df.to_csv('Yourdataname.csv')
```

  
