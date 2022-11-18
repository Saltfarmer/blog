---
title: "Pandas Exercise 4 : Apply"
header :
  teaser: /assets/images/pandas-head.jpg

categories:
  - Python
tags:
  - Pandas
  - Python
  - Exercise

---

The continuity of my practice on Pandas exercise from [guisapmora](https://github.com/guipsamora/pandas_exercises/archive/refs/heads/master.zip).

# United States - Crime Rates - 1960 - 2014

### Introduction:

This time you will create a data 

Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

### Step 1. Import the necessary libraries Dataset


```python
import pandas as pd
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/US_Crime_Rates/US_Crime_Rates_1960_2014.csv). 

### Step 3. Assign it to a variable called crime.


```python
url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/US_Crime_Rates/US_Crime_Rates_1960_2014.csv'
crime = pd.read_csv(url)
```

### Step 4. What is the type of the columns?


```python
crime.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 55 entries, 0 to 54
    Data columns (total 12 columns):
     #   Column              Non-Null Count  Dtype
    ---  ------              --------------  -----
     0   Year                55 non-null     int64
     1   Population          55 non-null     int64
     2   Total               55 non-null     int64
     3   Violent             55 non-null     int64
     4   Property            55 non-null     int64
     5   Murder              55 non-null     int64
     6   Forcible_Rape       55 non-null     int64
     7   Robbery             55 non-null     int64
     8   Aggravated_assault  55 non-null     int64
     9   Burglary            55 non-null     int64
     10  Larceny_Theft       55 non-null     int64
     11  Vehicle_Theft       55 non-null     int64
    dtypes: int64(12)
    memory usage: 5.3 KB
    

##### Have you noticed that the type of Year is int64. But pandas has a different type to work with Time Series. Let's see it now.

### Step 5. Convert the type of the column Year to datetime64


```python
crime['Year'] = pd.to_datetime(crime['Year'], format='%Y')
```

### Step 6. Set the Year column as the index of the dataframe


```python
crime.set_index(['Year'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Population</th>
      <th>Total</th>
      <th>Violent</th>
      <th>Property</th>
      <th>Murder</th>
      <th>Forcible_Rape</th>
      <th>Robbery</th>
      <th>Aggravated_assault</th>
      <th>Burglary</th>
      <th>Larceny_Theft</th>
      <th>Vehicle_Theft</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1960-01-01</th>
      <td>179323175</td>
      <td>3384200</td>
      <td>288460</td>
      <td>3095700</td>
      <td>9110</td>
      <td>17190</td>
      <td>107840</td>
      <td>154320</td>
      <td>912100</td>
      <td>1855400</td>
      <td>328200</td>
    </tr>
    <tr>
      <th>1961-01-01</th>
      <td>182992000</td>
      <td>3488000</td>
      <td>289390</td>
      <td>3198600</td>
      <td>8740</td>
      <td>17220</td>
      <td>106670</td>
      <td>156760</td>
      <td>949600</td>
      <td>1913000</td>
      <td>336000</td>
    </tr>
    <tr>
      <th>1962-01-01</th>
      <td>185771000</td>
      <td>3752200</td>
      <td>301510</td>
      <td>3450700</td>
      <td>8530</td>
      <td>17550</td>
      <td>110860</td>
      <td>164570</td>
      <td>994300</td>
      <td>2089600</td>
      <td>366800</td>
    </tr>
    <tr>
      <th>1963-01-01</th>
      <td>188483000</td>
      <td>4109500</td>
      <td>316970</td>
      <td>3792500</td>
      <td>8640</td>
      <td>17650</td>
      <td>116470</td>
      <td>174210</td>
      <td>1086400</td>
      <td>2297800</td>
      <td>408300</td>
    </tr>
    <tr>
      <th>1964-01-01</th>
      <td>191141000</td>
      <td>4564600</td>
      <td>364220</td>
      <td>4200400</td>
      <td>9360</td>
      <td>21420</td>
      <td>130390</td>
      <td>203050</td>
      <td>1213200</td>
      <td>2514400</td>
      <td>472800</td>
    </tr>
    <tr>
      <th>1965-01-01</th>
      <td>193526000</td>
      <td>4739400</td>
      <td>387390</td>
      <td>4352000</td>
      <td>9960</td>
      <td>23410</td>
      <td>138690</td>
      <td>215330</td>
      <td>1282500</td>
      <td>2572600</td>
      <td>496900</td>
    </tr>
    <tr>
      <th>1966-01-01</th>
      <td>195576000</td>
      <td>5223500</td>
      <td>430180</td>
      <td>4793300</td>
      <td>11040</td>
      <td>25820</td>
      <td>157990</td>
      <td>235330</td>
      <td>1410100</td>
      <td>2822000</td>
      <td>561200</td>
    </tr>
    <tr>
      <th>1967-01-01</th>
      <td>197457000</td>
      <td>5903400</td>
      <td>499930</td>
      <td>5403500</td>
      <td>12240</td>
      <td>27620</td>
      <td>202910</td>
      <td>257160</td>
      <td>1632100</td>
      <td>3111600</td>
      <td>659800</td>
    </tr>
    <tr>
      <th>1968-01-01</th>
      <td>199399000</td>
      <td>6720200</td>
      <td>595010</td>
      <td>6125200</td>
      <td>13800</td>
      <td>31670</td>
      <td>262840</td>
      <td>286700</td>
      <td>1858900</td>
      <td>3482700</td>
      <td>783600</td>
    </tr>
    <tr>
      <th>1969-01-01</th>
      <td>201385000</td>
      <td>7410900</td>
      <td>661870</td>
      <td>6749000</td>
      <td>14760</td>
      <td>37170</td>
      <td>298850</td>
      <td>311090</td>
      <td>1981900</td>
      <td>3888600</td>
      <td>878500</td>
    </tr>
    <tr>
      <th>1970-01-01</th>
      <td>203235298</td>
      <td>8098000</td>
      <td>738820</td>
      <td>7359200</td>
      <td>16000</td>
      <td>37990</td>
      <td>349860</td>
      <td>334970</td>
      <td>2205000</td>
      <td>4225800</td>
      <td>928400</td>
    </tr>
    <tr>
      <th>1971-01-01</th>
      <td>206212000</td>
      <td>8588200</td>
      <td>816500</td>
      <td>7771700</td>
      <td>17780</td>
      <td>42260</td>
      <td>387700</td>
      <td>368760</td>
      <td>2399300</td>
      <td>4424200</td>
      <td>948200</td>
    </tr>
    <tr>
      <th>1972-01-01</th>
      <td>208230000</td>
      <td>8248800</td>
      <td>834900</td>
      <td>7413900</td>
      <td>18670</td>
      <td>46850</td>
      <td>376290</td>
      <td>393090</td>
      <td>2375500</td>
      <td>4151200</td>
      <td>887200</td>
    </tr>
    <tr>
      <th>1973-01-01</th>
      <td>209851000</td>
      <td>8718100</td>
      <td>875910</td>
      <td>7842200</td>
      <td>19640</td>
      <td>51400</td>
      <td>384220</td>
      <td>420650</td>
      <td>2565500</td>
      <td>4347900</td>
      <td>928800</td>
    </tr>
    <tr>
      <th>1974-01-01</th>
      <td>211392000</td>
      <td>10253400</td>
      <td>974720</td>
      <td>9278700</td>
      <td>20710</td>
      <td>55400</td>
      <td>442400</td>
      <td>456210</td>
      <td>3039200</td>
      <td>5262500</td>
      <td>977100</td>
    </tr>
    <tr>
      <th>1975-01-01</th>
      <td>213124000</td>
      <td>11292400</td>
      <td>1039710</td>
      <td>10252700</td>
      <td>20510</td>
      <td>56090</td>
      <td>470500</td>
      <td>492620</td>
      <td>3265300</td>
      <td>5977700</td>
      <td>1009600</td>
    </tr>
    <tr>
      <th>1976-01-01</th>
      <td>214659000</td>
      <td>11349700</td>
      <td>1004210</td>
      <td>10345500</td>
      <td>18780</td>
      <td>57080</td>
      <td>427810</td>
      <td>500530</td>
      <td>3108700</td>
      <td>6270800</td>
      <td>966000</td>
    </tr>
    <tr>
      <th>1977-01-01</th>
      <td>216332000</td>
      <td>10984500</td>
      <td>1029580</td>
      <td>9955000</td>
      <td>19120</td>
      <td>63500</td>
      <td>412610</td>
      <td>534350</td>
      <td>3071500</td>
      <td>5905700</td>
      <td>977700</td>
    </tr>
    <tr>
      <th>1978-01-01</th>
      <td>218059000</td>
      <td>11209000</td>
      <td>1085550</td>
      <td>10123400</td>
      <td>19560</td>
      <td>67610</td>
      <td>426930</td>
      <td>571460</td>
      <td>3128300</td>
      <td>5991000</td>
      <td>1004100</td>
    </tr>
    <tr>
      <th>1979-01-01</th>
      <td>220099000</td>
      <td>12249500</td>
      <td>1208030</td>
      <td>11041500</td>
      <td>21460</td>
      <td>76390</td>
      <td>480700</td>
      <td>629480</td>
      <td>3327700</td>
      <td>6601000</td>
      <td>1112800</td>
    </tr>
    <tr>
      <th>1980-01-01</th>
      <td>225349264</td>
      <td>13408300</td>
      <td>1344520</td>
      <td>12063700</td>
      <td>23040</td>
      <td>82990</td>
      <td>565840</td>
      <td>672650</td>
      <td>3795200</td>
      <td>7136900</td>
      <td>1131700</td>
    </tr>
    <tr>
      <th>1981-01-01</th>
      <td>229146000</td>
      <td>13423800</td>
      <td>1361820</td>
      <td>12061900</td>
      <td>22520</td>
      <td>82500</td>
      <td>592910</td>
      <td>663900</td>
      <td>3779700</td>
      <td>7194400</td>
      <td>1087800</td>
    </tr>
    <tr>
      <th>1982-01-01</th>
      <td>231534000</td>
      <td>12974400</td>
      <td>1322390</td>
      <td>11652000</td>
      <td>21010</td>
      <td>78770</td>
      <td>553130</td>
      <td>669480</td>
      <td>3447100</td>
      <td>7142500</td>
      <td>1062400</td>
    </tr>
    <tr>
      <th>1983-01-01</th>
      <td>233981000</td>
      <td>12108600</td>
      <td>1258090</td>
      <td>10850500</td>
      <td>19310</td>
      <td>78920</td>
      <td>506570</td>
      <td>653290</td>
      <td>3129900</td>
      <td>6712800</td>
      <td>1007900</td>
    </tr>
    <tr>
      <th>1984-01-01</th>
      <td>236158000</td>
      <td>11881800</td>
      <td>1273280</td>
      <td>10608500</td>
      <td>18690</td>
      <td>84230</td>
      <td>485010</td>
      <td>685350</td>
      <td>2984400</td>
      <td>6591900</td>
      <td>1032200</td>
    </tr>
    <tr>
      <th>1985-01-01</th>
      <td>238740000</td>
      <td>12431400</td>
      <td>1328800</td>
      <td>11102600</td>
      <td>18980</td>
      <td>88670</td>
      <td>497870</td>
      <td>723250</td>
      <td>3073300</td>
      <td>6926400</td>
      <td>1102900</td>
    </tr>
    <tr>
      <th>1986-01-01</th>
      <td>240132887</td>
      <td>13211869</td>
      <td>1489169</td>
      <td>11722700</td>
      <td>20613</td>
      <td>91459</td>
      <td>542775</td>
      <td>834322</td>
      <td>3241410</td>
      <td>7257153</td>
      <td>1224137</td>
    </tr>
    <tr>
      <th>1987-01-01</th>
      <td>242282918</td>
      <td>13508700</td>
      <td>1483999</td>
      <td>12024700</td>
      <td>20096</td>
      <td>91110</td>
      <td>517704</td>
      <td>855088</td>
      <td>3236184</td>
      <td>7499900</td>
      <td>1288674</td>
    </tr>
    <tr>
      <th>1988-01-01</th>
      <td>245807000</td>
      <td>13923100</td>
      <td>1566220</td>
      <td>12356900</td>
      <td>20680</td>
      <td>92490</td>
      <td>542970</td>
      <td>910090</td>
      <td>3218100</td>
      <td>7705900</td>
      <td>1432900</td>
    </tr>
    <tr>
      <th>1989-01-01</th>
      <td>248239000</td>
      <td>14251400</td>
      <td>1646040</td>
      <td>12605400</td>
      <td>21500</td>
      <td>94500</td>
      <td>578330</td>
      <td>951710</td>
      <td>3168200</td>
      <td>7872400</td>
      <td>1564800</td>
    </tr>
    <tr>
      <th>1990-01-01</th>
      <td>248709873</td>
      <td>14475600</td>
      <td>1820130</td>
      <td>12655500</td>
      <td>23440</td>
      <td>102560</td>
      <td>639270</td>
      <td>1054860</td>
      <td>3073900</td>
      <td>7945700</td>
      <td>1635900</td>
    </tr>
    <tr>
      <th>1991-01-01</th>
      <td>252177000</td>
      <td>14872900</td>
      <td>1911770</td>
      <td>12961100</td>
      <td>24700</td>
      <td>106590</td>
      <td>687730</td>
      <td>1092740</td>
      <td>3157200</td>
      <td>8142200</td>
      <td>1661700</td>
    </tr>
    <tr>
      <th>1992-01-01</th>
      <td>255082000</td>
      <td>14438200</td>
      <td>1932270</td>
      <td>12505900</td>
      <td>23760</td>
      <td>109060</td>
      <td>672480</td>
      <td>1126970</td>
      <td>2979900</td>
      <td>7915200</td>
      <td>1610800</td>
    </tr>
    <tr>
      <th>1993-01-01</th>
      <td>257908000</td>
      <td>14144800</td>
      <td>1926020</td>
      <td>12218800</td>
      <td>24530</td>
      <td>106010</td>
      <td>659870</td>
      <td>1135610</td>
      <td>2834800</td>
      <td>7820900</td>
      <td>1563100</td>
    </tr>
    <tr>
      <th>1994-01-01</th>
      <td>260341000</td>
      <td>13989500</td>
      <td>1857670</td>
      <td>12131900</td>
      <td>23330</td>
      <td>102220</td>
      <td>618950</td>
      <td>1113180</td>
      <td>2712800</td>
      <td>7879800</td>
      <td>1539300</td>
    </tr>
    <tr>
      <th>1995-01-01</th>
      <td>262755000</td>
      <td>13862700</td>
      <td>1798790</td>
      <td>12063900</td>
      <td>21610</td>
      <td>97470</td>
      <td>580510</td>
      <td>1099210</td>
      <td>2593800</td>
      <td>7997700</td>
      <td>1472400</td>
    </tr>
    <tr>
      <th>1996-01-01</th>
      <td>265228572</td>
      <td>13493863</td>
      <td>1688540</td>
      <td>11805300</td>
      <td>19650</td>
      <td>96250</td>
      <td>535590</td>
      <td>1037050</td>
      <td>2506400</td>
      <td>7904700</td>
      <td>1394200</td>
    </tr>
    <tr>
      <th>1997-01-01</th>
      <td>267637000</td>
      <td>13194571</td>
      <td>1634770</td>
      <td>11558175</td>
      <td>18208</td>
      <td>96153</td>
      <td>498534</td>
      <td>1023201</td>
      <td>2460526</td>
      <td>7743760</td>
      <td>1354189</td>
    </tr>
    <tr>
      <th>1998-01-01</th>
      <td>270296000</td>
      <td>12475634</td>
      <td>1531044</td>
      <td>10944590</td>
      <td>16914</td>
      <td>93103</td>
      <td>446625</td>
      <td>974402</td>
      <td>2329950</td>
      <td>7373886</td>
      <td>1240754</td>
    </tr>
    <tr>
      <th>1999-01-01</th>
      <td>272690813</td>
      <td>11634378</td>
      <td>1426044</td>
      <td>10208334</td>
      <td>15522</td>
      <td>89411</td>
      <td>409371</td>
      <td>911740</td>
      <td>2100739</td>
      <td>6955520</td>
      <td>1152075</td>
    </tr>
    <tr>
      <th>2000-01-01</th>
      <td>281421906</td>
      <td>11608072</td>
      <td>1425486</td>
      <td>10182586</td>
      <td>15586</td>
      <td>90178</td>
      <td>408016</td>
      <td>911706</td>
      <td>2050992</td>
      <td>6971590</td>
      <td>1160002</td>
    </tr>
    <tr>
      <th>2001-01-01</th>
      <td>285317559</td>
      <td>11876669</td>
      <td>1439480</td>
      <td>10437480</td>
      <td>16037</td>
      <td>90863</td>
      <td>423557</td>
      <td>909023</td>
      <td>2116531</td>
      <td>7092267</td>
      <td>1228391</td>
    </tr>
    <tr>
      <th>2002-01-01</th>
      <td>287973924</td>
      <td>11878954</td>
      <td>1423677</td>
      <td>10455277</td>
      <td>16229</td>
      <td>95235</td>
      <td>420806</td>
      <td>891407</td>
      <td>2151252</td>
      <td>7057370</td>
      <td>1246646</td>
    </tr>
    <tr>
      <th>2003-01-01</th>
      <td>290690788</td>
      <td>11826538</td>
      <td>1383676</td>
      <td>10442862</td>
      <td>16528</td>
      <td>93883</td>
      <td>414235</td>
      <td>859030</td>
      <td>2154834</td>
      <td>7026802</td>
      <td>1261226</td>
    </tr>
    <tr>
      <th>2004-01-01</th>
      <td>293656842</td>
      <td>11679474</td>
      <td>1360088</td>
      <td>10319386</td>
      <td>16148</td>
      <td>95089</td>
      <td>401470</td>
      <td>847381</td>
      <td>2144446</td>
      <td>6937089</td>
      <td>1237851</td>
    </tr>
    <tr>
      <th>2005-01-01</th>
      <td>296507061</td>
      <td>11565499</td>
      <td>1390745</td>
      <td>10174754</td>
      <td>16740</td>
      <td>94347</td>
      <td>417438</td>
      <td>862220</td>
      <td>2155448</td>
      <td>6783447</td>
      <td>1235859</td>
    </tr>
    <tr>
      <th>2006-01-01</th>
      <td>299398484</td>
      <td>11401511</td>
      <td>1418043</td>
      <td>9983568</td>
      <td>17030</td>
      <td>92757</td>
      <td>447403</td>
      <td>860853</td>
      <td>2183746</td>
      <td>6607013</td>
      <td>1192809</td>
    </tr>
    <tr>
      <th>2007-01-01</th>
      <td>301621157</td>
      <td>11251828</td>
      <td>1408337</td>
      <td>9843481</td>
      <td>16929</td>
      <td>90427</td>
      <td>445125</td>
      <td>855856</td>
      <td>2176140</td>
      <td>6568572</td>
      <td>1095769</td>
    </tr>
    <tr>
      <th>2008-01-01</th>
      <td>304374846</td>
      <td>11160543</td>
      <td>1392628</td>
      <td>9767915</td>
      <td>16442</td>
      <td>90479</td>
      <td>443574</td>
      <td>842134</td>
      <td>2228474</td>
      <td>6588046</td>
      <td>958629</td>
    </tr>
    <tr>
      <th>2009-01-01</th>
      <td>307006550</td>
      <td>10762956</td>
      <td>1325896</td>
      <td>9337060</td>
      <td>15399</td>
      <td>89241</td>
      <td>408742</td>
      <td>812514</td>
      <td>2203313</td>
      <td>6338095</td>
      <td>795652</td>
    </tr>
    <tr>
      <th>2010-01-01</th>
      <td>309330219</td>
      <td>10363873</td>
      <td>1251248</td>
      <td>9112625</td>
      <td>14772</td>
      <td>85593</td>
      <td>369089</td>
      <td>781844</td>
      <td>2168457</td>
      <td>6204601</td>
      <td>739565</td>
    </tr>
    <tr>
      <th>2011-01-01</th>
      <td>311587816</td>
      <td>10258774</td>
      <td>1206031</td>
      <td>9052743</td>
      <td>14661</td>
      <td>84175</td>
      <td>354772</td>
      <td>752423</td>
      <td>2185140</td>
      <td>6151095</td>
      <td>716508</td>
    </tr>
    <tr>
      <th>2012-01-01</th>
      <td>313873685</td>
      <td>10219059</td>
      <td>1217067</td>
      <td>9001992</td>
      <td>14866</td>
      <td>85141</td>
      <td>355051</td>
      <td>762009</td>
      <td>2109932</td>
      <td>6168874</td>
      <td>723186</td>
    </tr>
    <tr>
      <th>2013-01-01</th>
      <td>316497531</td>
      <td>9850445</td>
      <td>1199684</td>
      <td>8650761</td>
      <td>14319</td>
      <td>82109</td>
      <td>345095</td>
      <td>726575</td>
      <td>1931835</td>
      <td>6018632</td>
      <td>700294</td>
    </tr>
    <tr>
      <th>2014-01-01</th>
      <td>318857056</td>
      <td>9475816</td>
      <td>1197987</td>
      <td>8277829</td>
      <td>14249</td>
      <td>84041</td>
      <td>325802</td>
      <td>741291</td>
      <td>1729806</td>
      <td>5858496</td>
      <td>689527</td>
    </tr>
  </tbody>
</table>
</div>



### Step 7. Delete the Total column


```python
crime = crime.drop(['Total'], axis=1)
```

### Step 8. Group the year by decades and sum the values

#### Pay attention to the Population column number, summing this column is a mistake


```python
crime.groupby((crime['Year'].dt.year // 10) * 10).sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Population</th>
      <th>Violent</th>
      <th>Property</th>
      <th>Murder</th>
      <th>Forcible_Rape</th>
      <th>Robbery</th>
      <th>Aggravated_assault</th>
      <th>Burglary</th>
      <th>Larceny_Theft</th>
      <th>Vehicle_Theft</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1960</th>
      <td>1915053175</td>
      <td>4134930</td>
      <td>45160900</td>
      <td>106180</td>
      <td>236720</td>
      <td>1633510</td>
      <td>2158520</td>
      <td>13321100</td>
      <td>26547700</td>
      <td>5292100</td>
    </tr>
    <tr>
      <th>1970</th>
      <td>2121193298</td>
      <td>9607930</td>
      <td>91383800</td>
      <td>192230</td>
      <td>554570</td>
      <td>4159020</td>
      <td>4702120</td>
      <td>28486000</td>
      <td>53157800</td>
      <td>9739900</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>2371370069</td>
      <td>14074328</td>
      <td>117048900</td>
      <td>206439</td>
      <td>865639</td>
      <td>5383109</td>
      <td>7619130</td>
      <td>33073494</td>
      <td>72040253</td>
      <td>11935411</td>
    </tr>
    <tr>
      <th>1990</th>
      <td>2612825258</td>
      <td>17527048</td>
      <td>119053499</td>
      <td>211664</td>
      <td>998827</td>
      <td>5748930</td>
      <td>10568963</td>
      <td>26750015</td>
      <td>77679366</td>
      <td>14624418</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>2947969117</td>
      <td>13968056</td>
      <td>100944369</td>
      <td>163068</td>
      <td>922499</td>
      <td>4230366</td>
      <td>8652124</td>
      <td>21565176</td>
      <td>67970291</td>
      <td>11412834</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>1570146307</td>
      <td>6072017</td>
      <td>44095950</td>
      <td>72867</td>
      <td>421059</td>
      <td>1749809</td>
      <td>3764142</td>
      <td>10125170</td>
      <td>30401698</td>
      <td>3569080</td>
    </tr>
  </tbody>
</table>
</div>



### Step 9. What is the most dangerous decade to live in the US?


```python
crime.groupby((crime['Year'].dt.year // 10) * 10).sum().\
drop(['Population'], axis=1).sum(axis=1).sort_values(ascending=False).head(1)
```




    Year
    1990    273162730
    dtype: int64


# Student Alcohol Consumption Dateset

### Introduction:

This time you will download a dataset from the UCI.

### Step 1. Import the necessary libraries


```python
import pandas as pd
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/Students_Alcohol_Consumption/student-mat.csv).

### Step 3. Assign it to a variable called df.


```python
url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/Students_Alcohol_Consumption/student-mat.csv'
df = pd.read_csv(url)
```

### Step 4. For the purpose of this exercise slice the dataframe from 'school' until the 'guardian' column


```python
df = df.loc[:, 'school':'guardian']
```

### Step 5. Create a lambda function that will capitalize strings.


```python
cap_fun = lambda x : x.capitalize()
```

### Step 6. Capitalize both Mjob and Fjob


```python
df['Mjob'].apply(cap_fun)
df['Fjob'].apply(cap_fun)
```




    0       Teacher
    1         Other
    2         Other
    3      Services
    4         Other
             ...   
    390    Services
    391    Services
    392       Other
    393       Other
    394     At_home
    Name: Fjob, Length: 395, dtype: object



### Step 7. Print the last elements of the data set.


```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>school</th>
      <th>sex</th>
      <th>age</th>
      <th>address</th>
      <th>famsize</th>
      <th>Pstatus</th>
      <th>Medu</th>
      <th>Fedu</th>
      <th>Mjob</th>
      <th>Fjob</th>
      <th>reason</th>
      <th>guardian</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>390</th>
      <td>MS</td>
      <td>M</td>
      <td>20</td>
      <td>U</td>
      <td>LE3</td>
      <td>A</td>
      <td>2</td>
      <td>2</td>
      <td>Services</td>
      <td>Services</td>
      <td>course</td>
      <td>other</td>
    </tr>
    <tr>
      <th>391</th>
      <td>MS</td>
      <td>M</td>
      <td>17</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>3</td>
      <td>1</td>
      <td>Services</td>
      <td>Services</td>
      <td>course</td>
      <td>mother</td>
    </tr>
    <tr>
      <th>392</th>
      <td>MS</td>
      <td>M</td>
      <td>21</td>
      <td>R</td>
      <td>GT3</td>
      <td>T</td>
      <td>1</td>
      <td>1</td>
      <td>Other</td>
      <td>Other</td>
      <td>course</td>
      <td>other</td>
    </tr>
    <tr>
      <th>393</th>
      <td>MS</td>
      <td>M</td>
      <td>18</td>
      <td>R</td>
      <td>LE3</td>
      <td>T</td>
      <td>3</td>
      <td>2</td>
      <td>Services</td>
      <td>Other</td>
      <td>course</td>
      <td>mother</td>
    </tr>
    <tr>
      <th>394</th>
      <td>MS</td>
      <td>M</td>
      <td>19</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>1</td>
      <td>1</td>
      <td>Other</td>
      <td>At_home</td>
      <td>course</td>
      <td>father</td>
    </tr>
  </tbody>
</table>
</div>



### Step 8. Did you notice the original dataframe is still lowercase? Why is that? Fix it and capitalize Mjob and Fjob.


```python
df['Mjob'] = df['Mjob'].apply(cap_fun)
df['Fjob'] = df['Fjob'].apply(cap_fun)
```

### Step 9. Create a function called majority that returns a boolean value to a new column called legal_drinker (Consider majority as older than 17 years old)


```python
majority = lambda x: True if(x > 17) else False
```


```python
df['legal_drinker'] = df['age'].apply(majority)
```

### Step 10. Multiply every number of the dataset by 10. 
##### I know this makes no sense, don't forget it is just an exercise


```python
df.apply(lambda x : x*10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>school</th>
      <th>sex</th>
      <th>age</th>
      <th>address</th>
      <th>famsize</th>
      <th>Pstatus</th>
      <th>Medu</th>
      <th>Fedu</th>
      <th>Mjob</th>
      <th>Fjob</th>
      <th>reason</th>
      <th>guardian</th>
      <th>legal_drinker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GPGPGPGPGPGPGPGPGPGP</td>
      <td>FFFFFFFFFF</td>
      <td>180</td>
      <td>UUUUUUUUUU</td>
      <td>GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3</td>
      <td>AAAAAAAAAA</td>
      <td>40</td>
      <td>40</td>
      <td>At_homeAt_homeAt_homeAt_homeAt_homeAt_homeAt_h...</td>
      <td>TeacherTeacherTeacherTeacherTeacherTeacherTeac...</td>
      <td>coursecoursecoursecoursecoursecoursecoursecour...</td>
      <td>mothermothermothermothermothermothermothermoth...</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GPGPGPGPGPGPGPGPGPGP</td>
      <td>FFFFFFFFFF</td>
      <td>170</td>
      <td>UUUUUUUUUU</td>
      <td>GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3</td>
      <td>TTTTTTTTTT</td>
      <td>10</td>
      <td>10</td>
      <td>At_homeAt_homeAt_homeAt_homeAt_homeAt_homeAt_h...</td>
      <td>OtherOtherOtherOtherOtherOtherOtherOtherOtherO...</td>
      <td>coursecoursecoursecoursecoursecoursecoursecour...</td>
      <td>fatherfatherfatherfatherfatherfatherfatherfath...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GPGPGPGPGPGPGPGPGPGP</td>
      <td>FFFFFFFFFF</td>
      <td>150</td>
      <td>UUUUUUUUUU</td>
      <td>LE3LE3LE3LE3LE3LE3LE3LE3LE3LE3</td>
      <td>TTTTTTTTTT</td>
      <td>10</td>
      <td>10</td>
      <td>At_homeAt_homeAt_homeAt_homeAt_homeAt_homeAt_h...</td>
      <td>OtherOtherOtherOtherOtherOtherOtherOtherOtherO...</td>
      <td>otherotherotherotherotherotherotherotherothero...</td>
      <td>mothermothermothermothermothermothermothermoth...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GPGPGPGPGPGPGPGPGPGP</td>
      <td>FFFFFFFFFF</td>
      <td>150</td>
      <td>UUUUUUUUUU</td>
      <td>GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3</td>
      <td>TTTTTTTTTT</td>
      <td>40</td>
      <td>20</td>
      <td>HealthHealthHealthHealthHealthHealthHealthHeal...</td>
      <td>ServicesServicesServicesServicesServicesServic...</td>
      <td>homehomehomehomehomehomehomehomehomehome</td>
      <td>mothermothermothermothermothermothermothermoth...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GPGPGPGPGPGPGPGPGPGP</td>
      <td>FFFFFFFFFF</td>
      <td>160</td>
      <td>UUUUUUUUUU</td>
      <td>GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3</td>
      <td>TTTTTTTTTT</td>
      <td>30</td>
      <td>30</td>
      <td>OtherOtherOtherOtherOtherOtherOtherOtherOtherO...</td>
      <td>OtherOtherOtherOtherOtherOtherOtherOtherOtherO...</td>
      <td>homehomehomehomehomehomehomehomehomehome</td>
      <td>fatherfatherfatherfatherfatherfatherfatherfath...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>390</th>
      <td>MSMSMSMSMSMSMSMSMSMS</td>
      <td>MMMMMMMMMM</td>
      <td>200</td>
      <td>UUUUUUUUUU</td>
      <td>LE3LE3LE3LE3LE3LE3LE3LE3LE3LE3</td>
      <td>AAAAAAAAAA</td>
      <td>20</td>
      <td>20</td>
      <td>ServicesServicesServicesServicesServicesServic...</td>
      <td>ServicesServicesServicesServicesServicesServic...</td>
      <td>coursecoursecoursecoursecoursecoursecoursecour...</td>
      <td>otherotherotherotherotherotherotherotherothero...</td>
      <td>10</td>
    </tr>
    <tr>
      <th>391</th>
      <td>MSMSMSMSMSMSMSMSMSMS</td>
      <td>MMMMMMMMMM</td>
      <td>170</td>
      <td>UUUUUUUUUU</td>
      <td>LE3LE3LE3LE3LE3LE3LE3LE3LE3LE3</td>
      <td>TTTTTTTTTT</td>
      <td>30</td>
      <td>10</td>
      <td>ServicesServicesServicesServicesServicesServic...</td>
      <td>ServicesServicesServicesServicesServicesServic...</td>
      <td>coursecoursecoursecoursecoursecoursecoursecour...</td>
      <td>mothermothermothermothermothermothermothermoth...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>392</th>
      <td>MSMSMSMSMSMSMSMSMSMS</td>
      <td>MMMMMMMMMM</td>
      <td>210</td>
      <td>RRRRRRRRRR</td>
      <td>GT3GT3GT3GT3GT3GT3GT3GT3GT3GT3</td>
      <td>TTTTTTTTTT</td>
      <td>10</td>
      <td>10</td>
      <td>OtherOtherOtherOtherOtherOtherOtherOtherOtherO...</td>
      <td>OtherOtherOtherOtherOtherOtherOtherOtherOtherO...</td>
      <td>coursecoursecoursecoursecoursecoursecoursecour...</td>
      <td>otherotherotherotherotherotherotherotherothero...</td>
      <td>10</td>
    </tr>
    <tr>
      <th>393</th>
      <td>MSMSMSMSMSMSMSMSMSMS</td>
      <td>MMMMMMMMMM</td>
      <td>180</td>
      <td>RRRRRRRRRR</td>
      <td>LE3LE3LE3LE3LE3LE3LE3LE3LE3LE3</td>
      <td>TTTTTTTTTT</td>
      <td>30</td>
      <td>20</td>
      <td>ServicesServicesServicesServicesServicesServic...</td>
      <td>OtherOtherOtherOtherOtherOtherOtherOtherOtherO...</td>
      <td>coursecoursecoursecoursecoursecoursecoursecour...</td>
      <td>mothermothermothermothermothermothermothermoth...</td>
      <td>10</td>
    </tr>
    <tr>
      <th>394</th>
      <td>MSMSMSMSMSMSMSMSMSMS</td>
      <td>MMMMMMMMMM</td>
      <td>190</td>
      <td>UUUUUUUUUU</td>
      <td>LE3LE3LE3LE3LE3LE3LE3LE3LE3LE3</td>
      <td>TTTTTTTTTT</td>
      <td>10</td>
      <td>10</td>
      <td>OtherOtherOtherOtherOtherOtherOtherOtherOtherO...</td>
      <td>At_homeAt_homeAt_homeAt_homeAt_homeAt_homeAt_h...</td>
      <td>coursecoursecoursecoursecoursecoursecoursecour...</td>
      <td>fatherfatherfatherfatherfatherfatherfatherfath...</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
<p>395 rows × 13 columns</p>
</div>


