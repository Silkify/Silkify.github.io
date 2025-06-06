---
layout: single
title:  "Online Retail II Cohort Analysis"
categories: 
  - Project
toc: true
author_profile: True
permalink: /categories/Project/Project-2-1
---

# Introduction

## Background(배경)

The ability to analyze user behavior over time through data is a core skill that every analyst should possess, as it plays a crucial role in uncovering meaningful insights. <br>
Cohort analysis is one such method that helps in effectively tracking and understanding changes in user behavior across different groups.

시간에 따른 사용자의 행동을 데이터로 분석하는 능력은 분석가가 반드시 갖추어야 할 핵심 역량이며, 이는 의미 있는 인사이트를 도출하는 데 큰 도움이 됩니다. <br>
이러한 분석을 효과적으로 수행하기 위한 방법 중 하나가 바로 코호트 분석이며, 사용자 그룹의 행동 변화를 추적하고 이해하는 데 유용한 도구로 활용됩니다.

## What is Cohort?

Cohort meaning is group of individuals who share common characters within specific time period <br>
코호트란 특정 시점 또는 기간에 동일한 특성을 가진 집단을 의미합니다.

## Using Data

Online Retail II <br>
link : "https://archive.ics.uci.edu/dataset/502/online+retail+ii"

## Using Packages


```python
import pandas as pd
import numpy as np
from plotnine import *
from scipy.stats import chi2_contingency
from scipy.stats import chi2_contingency, mannwhitneyu
```


```python
df_2010 = pd.read_csv('csvfile1.csv') # 2009 - 2010
df_2011 = pd.read_csv('csvfile.csv') # 2010 - 2011
```

## Preprocessing Before Analysis (Handling Data Set)

### Handling Missing Data <결측치 처리>

1. If a Customer ID is missing, it may indicate that the user did not register 

    e.g., in cases where a guest makes a purchase.

    Customer ID가 없는 경우 등록하지 않았을 수도 있다고 생각
    
    ex) 비회원이 구매하는 경우

2. Upon examining the InvoiceDate, which records the purchase time down to the minute,

    we found instances where multiple entries had the exact same purchase time but different Customer IDs, with most of the other information missing.
    
    We interpreted this as likely representing the same individual and handled the data accordingly.
    
    InvoiceDate 구매 시간 분단위 기록을 살펴보면 구매시간이 동일하고 Custer ID가 다른 경우는 결측 값 밖에 없는 것을 파악하여 이는 동일한 사람이라고 판단하여 처리.

3. 상품명이 없는 경우 제외처리 

   Specifically, rows where the Description field is NaN were excluded from the analysis.


```python
invoice_to_customer = (
    df_2010[~df_2010['Customer ID'].isna()]
    .groupby('InvoiceDate')['Customer ID']
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
)

invoice_to_customer1 = (
    df_2011[~df_2011['Customer ID'].isna()]
    .groupby('InvoiceDate')['Customer ID']
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
)
```


```python
def fill_customer_id(row):
    if pd.isna(row['Customer ID']):
        match_id = invoice_to_customer.get(row['InvoiceDate'], None)
        return match_id if pd.notna(match_id) else None
    return row['Customer ID']
```


```python
df_2010['Customer ID'] = df_2010.apply(fill_customer_id, axis=1)
```


```python
df_2011['Customer ID'] = df_2011.apply(fill_customer_id, axis=1)
```


```python
na_2010 = df_2010[df_2010['Customer ID'].isna()]
na_2011 = df_2011[df_2011['Customer ID'].isna()]

df_2010 = df_2010.drop(na_2010.index)
df_2011 = df_2011.drop(na_2011.index)
```


```python
df_2010.dropna(subset='Description',inplace = True)
df_2011.dropna(subset='Description',inplace = True)
```


```python
df_2010['InvoiceDate'] = pd.to_datetime(df_2010['InvoiceDate'])
df_2011['InvoiceDate'] = pd.to_datetime(df_2011['InvoiceDate'])

df = pd.concat([df_2010,df_2011])
```


```python
# 날짜를 day 기준으로 변환
df_2010['InvoiceDay'] = df_2010['InvoiceDate'].dt.to_period('D')
df_2011['InvoiceDay'] = df_2011['InvoiceDate'].dt.to_period('D')
df['InvoiceDay'] = df['InvoiceDate'].dt.to_period('D')


# Cohort 기준일 설정 (가입일 또는 첫 구매일)
df_2010['CohortDay'] = df_2010.groupby('Customer ID')['InvoiceDate'].transform('min').dt.to_period('D')
df_2011['CohortDay'] = df_2011.groupby('Customer ID')['InvoiceDate'].transform('min').dt.to_period('D')
df_2010['CohortMonth'] = df_2010.groupby('Customer ID')['InvoiceDate'].transform('min').dt.to_period('M')
df_2011['CohortMonth'] = df_2011.groupby('Customer ID')['InvoiceDate'].transform('min').dt.to_period('M')
df['CohortDay'] = df.groupby('Customer ID')['InvoiceDate'].transform('min').dt.to_period('D')
df['CohortMonth'] = df.groupby('Customer ID')['InvoiceDate'].transform('min').dt.to_period('M')


# CohortIndex 계산 (며칠째 방문인지)
df_2010['CohortDIndex'] = (df_2010['InvoiceDay'] - df_2010['CohortDay']).apply(lambda x: x.n)
df_2011['CohortDIndex'] = (df_2011['InvoiceDay'] - df_2011['CohortDay']).apply(lambda x: x.n)
df['CohortDIndex'] = (df['InvoiceDay'] - df['CohortDay']).apply(lambda x: x.n)


df_2010['CohortMIndex'] = (df_2010['InvoiceMonth'] - df_2010['CohortMonth']).apply(lambda x: x.n)
df_2011['CohortMIndex'] = (df_2011['InvoiceMonth'] - df_2011['CohortMonth']).apply(lambda x: x.n)
df['CohortMIndex'] = (df['InvoiceMonth'] - df['CohortMonth']).apply(lambda x: x.n)
```


```python

```


```python
cohort_data = df_2010.groupby(['CohortMonth', 'CohortMIndex'])['Customer ID'].nunique().unstack(0)
cohort_data1 = df.groupby(['CohortMonth', 'CohortMIndex'])['Customer ID'].nunique().unstack(0)
```


```python
cohort_size = cohort_data.loc[0]
cohort_size1 = cohort_data1.loc[0]
```


```python
retention = cohort_data.divide(cohort_size, axis=1)
retention1 = cohort_data1.divide(cohort_size1, axis=1)
```


# Date Anaylsis

## Retention graph



```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.heatmap(retention1.T, annot=True, fmt='.0%', cmap='YlGnBu')
plt.title('Cohort Retention Rate')
plt.xlabel('Cohort Index (Months)')
plt.ylabel('Cohort Month')
plt.show()
```


    
![png](/CohortRetailshop/output_22_0.png)
    


1. Customer Chunrate is high during first three Month

   3개월 동안 고객 이탈율이 높다.

2. After 3 month, Customer Chunrate drops significantly, I think that customers become repeat buyers.

    4개월부터 고객 이탈율이 매우 낮아지고 단골 손님이 되는 것 같다.
  
3. I think There appears to be a recurring event, such as Black Friday, between October and November each year, as customer retention spikes noticeably during this period.

   매년 10~11월사이에 특정한 이벤트(ex. 블랙프라이데이)와 같은 것을 하는 것으로 추정 된다. 급격하게 증가하는 것을 볼 수 있음.


```python
cohort_revenue = df.groupby(['CohortMonth', 'CohortIndex'])['TotalPrice'].sum().unstack(0)
```


```python
cohort_first_revenue = cohort_revenue.iloc[0]  # 각 코호트의 첫달 매출
revenue_ratio = cohort_revenue.divide(cohort_first_revenue,axis=1)
```


```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.heatmap(revenue_ratio, annot=True, fmt=".2f", cmap="Blues")
plt.title('Cohort Revenue Ratio (Compared to First Month)')
plt.ylabel('Months Since First Purchase (CohortIndex)')
plt.xlabel('Cohort Month')
plt.show()
```


    
![png](/CohortRetailshop/output_26_0.png)
    



```python
# 고객 수 (CohortMonth, CohortIndex별)
cohort_sizes = df.groupby(['CohortMonth', 'CohortIndex'])['Customer ID'].nunique().unstack(0)

# 매출 합계 (CohortMonth, CohortIndex별)
cohort_revenue = df.groupby(['CohortMonth', 'CohortIndex'])['TotalPrice'].sum().unstack(0)
```


```python
cohort_ltv = cohort_revenue / cohort_sizes
cumulative_ltv = cohort_ltv.cumsum()
import matplotlib.pyplot as plt

palette = sns.color_palette('tab20', n_colors=len(cumulative_ltv.columns))
plt.figure(figsize=(14, 8))
for i, cohort in enumerate(cumulative_ltv.columns):
    plt.plot(cumulative_ltv.index, cumulative_ltv[cohort], marker='o', label=str(cohort), color=palette[i])

plt.title('Cohort Cumulative LTV (Average Revenue per Customer)')
plt.xlabel('Months Since First Purchase (CohortIndex)')
plt.ylabel('Cumulative LTV')
plt.legend(title='Cohort Month')
plt.grid(True)
plt.show()
```


    
![png](/CohortRetailshop/output_28_0.png)
    


Data from December 2009 is too unique to be useful. <br>
Because it marks the beginning of the dataset and possibly the start of an event, the total sales volume for that month appears as an outlier.

- 2009-12는 조금 특이한 데이터이다. 데이터의 시작이기도 하고 이벤트의 시작이 있었는지 12월 당월의 총 판매량은 이상치라고 느낄 정도로 매우 크다.

Therefore, I thought to exclude that month when analyzing the data. By doing so, I found that June 2010 and September 2010 show significantly higher values compared to other months.
- 따라서 본인은 데이터를 살펴 볼 때 2009-12를 제외하고 살펴보기로 하였고 2010-06과 2010-09가 다른 월에 비하여 매우 높다고 생각하여 이를 중점으로 살펴볼까 한다.


```python
cohort_201006 = df[df['CohortMonth'] == pd.Period('2010-06')]
cohort_201009 = df[df['CohortMonth'] == pd.Period('2010-09')]
```


```python
def cohort_summary(cohort_df):
    summary = cohort_df.groupby('CohortIndex').agg(
        customers=('Customer ID', 'nunique'),
        total_revenue=('TotalPrice', 'sum')
    )
    summary['avg_revenue_per_customer'] = summary['total_revenue'] / summary['customers']
    return summary

summary_201006 = cohort_summary(cohort_201006)
summary_201009 = cohort_summary(cohort_201009)
```


```python
summary_201006_no0 = summary_201006[summary_201006.index != 0]
summary_201009_no0 = summary_201009[summary_201009.index != 0]
```


```python
import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))

plt.plot(summary_201006_no0.index, summary_201006_no0['avg_revenue_per_customer'], marker='o', label='2010-06 Cohort')

plt.plot(summary_201009_no0.index + 3, summary_201009_no0['avg_revenue_per_customer'], marker='o', label='2010-09 Cohort')

plt.title('Average Revenue per Customer by CohortIndex (3-month offset applied)')
plt.xlabel('Months Since First Purchase (CohortIndex with offset)')
plt.ylabel('Average Revenue per Customer')
plt.legend()
plt.grid(True)


max_index = max(summary_201006_no0.index.max(), summary_201009_no0.index.max() + 3)
plt.xticks(ticks=range(0, max_index + 1))

plt.show()
```


    
![png](/CohortRetailshop/output_33_0.png)
    


Possibility 1: The two cohorts may prefer different products.

가능성 1: 두 코호트가 선호하는 물품이 다를 가능성

Possibility 2: The two cohorts may have purchased the same products, but the difference is simply due to issues such as expiration dates.

가능성 2: 두 코호트가 같은 물품을 샀으나 유통기한과 같은 문제로 단순 차이일 뿐이다.

Possibility 3: The volatility may be high because customers who made bulk purchases were included.

가능성 3: 대랑구매를 했던 고객이 들어와서 변동성이 클 수 있다.

## 두 집단의 재구매율 (2개월 이상 재구매 비율)


```python
def repurchase_rate(cohort_df):
    purchase_counts = cohort_df.groupby('Customer ID')['CohortIndex'].nunique()
    repurchase_ratio = (purchase_counts >= 2).mean()
    return repurchase_ratio

repurchase_201006 = repurchase_rate(cohort_201006)
repurchase_201009 = repurchase_rate(cohort_201009)

print(f"2010-06 Cohort Re-purchase Rate (>=2 months): {repurchase_201006:.2%}")
print(f"2010-09 Cohort Re-purchase Rate (>=2 months): {repurchase_201009:.2%}")
```

    2010-06 Cohort Re-purchase Rate (>=2 months): 72.12%
    2010-09 Cohort Re-purchase Rate (>=2 months): 71.07%
    

그래프로도 볼 수 있는 내용이지만 수치적으로도 확실히 비율이 높다는 것을 알 수 있다.

## Consideration of Possibility 1,2,3

### 가설
- 영가설 (H₀): 두 코호트(2010-06과 2010-09)의 선호하는 제품 종류는 같다.
- 대안가설 (H₁): 두 코호트의 선호하는 제품 종류는 다르다.

하지만 이것을 하기 전에 가격차이,수량차이 등 따져볼 것이 있다.


```python
df_1006 = df[df['CohortMonth'] == '2010-06']
df_1009 = df[df['CohortMonth'] == '2010-09']
```


```python
purchase_1006 = df_1006.groupby('Description')[['Quantity','TotalPrice']].sum()
purchase_1006['Price'] = np.round(purchase_1006['TotalPrice']/purchase_1006['Quantity'],2)

purchase_1009 = df_1009.groupby('Description')[['Quantity','TotalPrice']].sum()
purchase_1009['Price'] = np.round(purchase_1009['TotalPrice']/purchase_1009['Quantity'],2)
```

- 두 코호트 사이의 가격의 차이가 있는가?


```python
from scipy.stats import mannwhitneyu

price_06 = purchase_1006['Price'].dropna()
price_09 = purchase_1009['Price'].dropna()

stat, p_price = mannwhitneyu(price_06, price_09, alternative='two-sided')
print(f"Price distribution Mann-Whitney U p-value: {p_price:.4f}")

```

    Price distribution Mann-Whitney U p-value: 0.1624
    

- 구매 수량이 차이가 있는가?


```python
cust_qty_06 = df_1006.groupby('Customer ID')['Quantity'].sum()
cust_qty_09 = df_1009.groupby('Customer ID')['Quantity'].sum()

from scipy.stats import mannwhitneyu
stat, p = mannwhitneyu(cust_qty_06, cust_qty_09, alternative='two-sided')
print(f"수량 차이 Mann-Whitney U 검정 p-value: {p:.4f}")
```

    수량 차이 Mann-Whitney U 검정 p-value: 0.9890
    


```python
top_desc = pd.concat([purchase_1006['Quantity'], purchase_1009['Quantity']], axis=1).fillna(0)
top_desc['Total'] = top_desc.sum(axis=1)
top_50_desc = top_desc.sort_values('Total', ascending=False).head(50).index

qty_06 = purchase_1006.reindex(top_50_desc, fill_value=0)['Quantity']
qty_09 = purchase_1009.reindex(top_50_desc, fill_value=0)['Quantity']

contingency_table = pd.DataFrame({
    '2010-06': qty_06,
    '2010-09': qty_09
})

chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"Chi2 Statistic: {chi2:.4f}")
print(f"p-value: {p:.4f}")
```

    Chi2 Statistic: 120130.8214
    p-value: 0.0000
    

### 결론

| 분석 항목             | 검정 방법          | p-value  | 해석                      |
|---------------------|------------------|----------|-------------------------|
| 가격 차이             | Mann-Whitney U   | 0.1624   | 차이 없다                 |
| 고객별 구매 수량 차이    | Mann-Whitney U   | 0.9890   | 차이 없다                 |
| 제품별 구매 수량 분포 차이 | 카이제곱 검정      | 0.0000   | 차이가 있다 (선호도 다름)   |


P-value가 0.05보다 작기 때문에 두 코호트간의 선호도 차이는 존재한다고 할 수 있다.

## RFM Analysis

**R**ecency **F**requency **M**onetary
(구매시점,빈도,금액)
- 이 세가지 지표로 고객의 충성도를 분류

> 1. **Recency**

> 1주를 기준으로 점수를 부여 ex) **1주차** 5/**(1)**, **2주차** 5/**(2)** ...

> 2. **Frequency**

> 5분위수를 기준으로 1~5점으로 점수 부여

> 3. **Monetary**

> **0~50** : 1점 , **51~100** : 2점, **100~200** : 3점, **200~300** : 4점, **300 이상** : 5점



```python
import datetime
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'Invoice': 'nunique',                                    # Frequency (unique invoice count)
    'TotalPrice': 'sum'                                      # Monetary
}).rename(columns={
    'InvoiceDate': 'Recency',
    'Invoice': 'Frequency',
    'TotalPrice': 'Monetary'
})
```


```python
def recency_score(days):
    score = 5 - (days // 7)
    return max(score, 1)

rfm['R_score'] = rfm['Recency'].apply(recency_score)

quantiles = rfm['Frequency'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).values

def frequency_score_manual(x):
    if x <= quantiles[1]:
        return 1
    elif x <= quantiles[2]:
        return 2
    elif x <= quantiles[3]:
        return 3
    elif x <= quantiles[4]:
        return 4
    else:
        return 5

rfm['F_score'] = rfm['Frequency'].apply(frequency_score_manual)

def monetary_score(x):
    if x <= 50:
        return 1
    elif x <= 100:
        return 2
    elif x <= 200:
        return 3
    elif x <= 300:
        return 4
    else:
        return 5

rfm['M_score'] = rfm['Monetary'].apply(monetary_score)


rfm['F_score'] = rfm['F_score'].astype(int)
rfm['R_score'] = rfm['R_score'].astype(int)
rfm['M_score'] = rfm['M_score'].astype(int)
```


```python
rfm['RFM Score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']
```


```python
def rfm_level(score):
    if score <= 5:
        return 'Low'
    elif score <= 10:
        return 'Medium'
    else:
        return 'High'

rfm['RFM_level'] = rfm['RFM Score'].apply(rfm_level)
```


```python
_rfm = rfm.reset_index()
```


```python
_dummy = _rfm[['Customer ID','RFM_level']]
```


```python
dummy_df = pd.merge(df,_dummy)
```


```python
rfm_level_month = dummy_df.groupby(['CohortMonth','RFM_level']).size().reset_index(name='count')
```


```python
rfm_level_month['RFM_level'] = pd.Categorical(rfm_level_month['RFM_level'],
                                       categories=['High', 'Medium', 'Low'],
                                       ordered=True)
```


```python
pivot_df = rfm_level_month.pivot_table(index='CohortMonth', columns='RFM_level', values='count', aggfunc='sum')
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_df, annot=True, fmt='g', cmap='YlGnBu')
plt.title('Count by CohortMonth and RFM Level')
plt.ylabel('Cohort Month')
plt.xlabel('RFM Level')
plt.show()
```

    C:\Users\Administrator\AppData\Local\Temp\ipykernel_4024\231497128.py:1: FutureWarning: The default value of observed=False is deprecated and will change to observed=True in a future version of pandas. Specify observed=False to silence this warning and retain the current behavior
    


    
![png](/CohortRetailshop/output_61_1.png)
    



```python
import numpy as np

plt.figure(figsize=(10, 6))


ax = sns.heatmap(pivot_df, annot=True, fmt='g', cmap='YlGnBu')

highlight_rows = ['2010-06', '2010-09']
for row_label in highlight_rows:
    if row_label in pivot_df.index:
        row_idx = pivot_df.index.get_loc(row_label)
        ax.hlines(row_idx, *ax.get_xlim(), colors='red', linewidth=3)
        ax.hlines(row_idx+1, *ax.get_xlim(), colors='red', linewidth=3)

plt.title('Count by CohortMonth and RFM Level')
plt.ylabel('Cohort Month')
plt.xlabel('RFM Level')
plt.show()
```


    
![png](/CohortRetailshop/output_62_0.png)
    


#### 2010-06 & 2010-09 differnce of RFM
- 2010-06

>Size : 27756

>High: 12,527 (45.1% of total)

>Medium: 14,739 (53.1%)

>Low: 499 (1.8%)
- 2010-09
  
>Size : 21200

>High: 11,131 (52.5% of total)

>Medium: 9,739 (45.9%)

>Low: 330 (1.6%)

2010-06의 코호트가 집단의 크기가 더 크다는 것을 알 수 있고 중간층의 충성도가 높음을 알 수 있다. 

2010-09의 코호트는 장기적으로 보았을 때 높은 충성도를 가진 고객이 많은 것을 알 수 있다.

`-` 단! 위의 히트맵은 Customer ID로 그룹하지 않은거니 조금 더 살펴봐야 할 것같다!


```python
df_1006 = dummy_df[dummy_df['CohortMonth'] == '2010-06']
df_1009 = dummy_df[dummy_df['CohortMonth'] == '2010-09']
```


## Marketing Strategy

## Customer ID RFM LEVEL


```python
rfm_level_count = dummy_df.groupby(['CohortMonth', 'RFM_level'])['Customer ID'].nunique().reset_index()
rfm_level_count.rename(columns={'Customer ID': 'CustomerCount'}, inplace=True)

rfm_level_count['RFM_level'] = pd.Categorical(rfm_level_count['RFM_level'],
                                       categories=['High', 'Medium', 'Low'],
                                       ordered=True)
pivot_rfm = rfm_level_count.pivot(index='CohortMonth', columns='RFM_level', values='CustomerCount')

plt.figure(figsize=(12, 6))
sns.heatmap(pivot_rfm, annot=True, fmt='g', cmap='YlGnBu', linewidths=0.5)
plt.title('CohortMonth × RFM_level Customer Heatmap')
plt.ylabel('CohortMonth')
plt.xlabel('RFM_level')
plt.show()
```


    
![png](/CohortRetailshop/output_67_0.png)
    


각 코호트 집단별로 제품을 10개씩 뽑아서 이들이 자주 구매했던 제품들을 찾아 낼 필요가 있다. (코드는 3개로 하였는데 이는 24개월치이기 때문에 너무 많기 때문이다.)


```python
top_n = 3

cohort_list = dummy_df['CohortMonth'].unique()
rfm_level_list = ['Low', 'Medium', 'High']

for cohort in sorted(cohort_list):
    for level in rfm_level_list:
        subset = dummy_df[(dummy_df['CohortMonth'] == cohort) & (dummy_df['RFM_level'] == level)]
        top_products = subset.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(top_n)
        print(f"\n--- {cohort} Cohort | RFM Level: {level} ---")
        print(top_products)
```

    
    --- 2009-12 Cohort | RFM Level: Low ---
    Description
    CHINESE DRAGON PAPER LANTERNS    168
    JACOBS LADDER SMALL              144
    JUMBO BAG RED WHITE SPOTTY       110
    Name: Quantity, dtype: int64
    
    --- 2009-12 Cohort | RFM Level: Medium ---
    Description
    SMALL CHINESE STYLE SCISSOR           26293
    BLACK AND WHITE PAISLEY FLOWER MUG    25200
    SET/6 WOODLAND PAPER PLATES           12964
    Name: Quantity, dtype: int64
    
    --- 2009-12 Cohort | RFM Level: High ---
    Description
    BROCADE RING PURSE                    50973
    WHITE HANGING HEART T-LIGHT HOLDER    49759
    ASSORTED COLOUR BIRD ORNAMENT         47232
    Name: Quantity, dtype: int64
    
    --- 2010-01 Cohort | RFM Level: Low ---
    Description
    ANTIQUE SILVER TEA GLASS ENGRAVED    72
    CACTI T-LIGHT CANDLES                48
    PACK OF 12 HEARTS DESIGN TISSUES     48
    Name: Quantity, dtype: int64
    
    --- 2010-01 Cohort | RFM Level: Medium ---
    Description
    ASSORTED LAQUERED INCENSE HOLDERS     4332
    ASSORTED FLOWER COLOUR "LEIS"         2754
    WHITE HANGING HEART T-LIGHT HOLDER    2388
    Name: Quantity, dtype: int64
    
    --- 2010-01 Cohort | RFM Level: High ---
    Description
    60 TEATIME FAIRY CAKE CASES           11315
    PACK OF 60 PINK PAISLEY CAKE CASES     9837
    PACK OF 72 SKULL CAKE CASES            9153
    Name: Quantity, dtype: int64
    
    --- 2010-02 Cohort | RFM Level: Low ---
    Description
    ANTIQUE SILVER TEA GLASS ENGRAVED    78
    KEY FOB , FRONT  DOOR                48
    KEY FOB , BACK DOOR                  48
    Name: Quantity, dtype: int64
    
    --- 2010-02 Cohort | RFM Level: Medium ---
    Description
    WHITE HANGING HEART T-LIGHT HOLDER    2223
    INLAID WOOD INCENSE HOLDER            1987
    ASSTD DESIGN BUBBLE GUM RING          1500
    Name: Quantity, dtype: int64
    
    --- 2010-02 Cohort | RFM Level: High ---
    Description
    JUMBO BAG RED RETROSPOT                6070
    MINI HIGHLIGHTER PENS                  5880
    ESSENTIAL BALM 3.5g TIN IN ENVELOPE    5590
    Name: Quantity, dtype: int64
    
    --- 2010-03 Cohort | RFM Level: Low ---
    Description
    CACTI T-LIGHT CANDLES             48
    JAM JAR WITH GREEN LID            48
    FRENCH STYLE STORAGE JAR CAFE     48
    Name: Quantity, dtype: int64
    
    --- 2010-03 Cohort | RFM Level: Medium ---
    Description
    ASSORTED COLOURS SILK FAN            3204
    GIRLS ALPHABET IRON ON PATCHES       2592
    WORLD WAR 2 GLIDERS ASSTD DESIGNS    2064
    Name: Quantity, dtype: int64
    
    --- 2010-03 Cohort | RFM Level: High ---
    Description
    WORLD WAR 2 GLIDERS ASSTD DESIGNS    26320
    RED  HARMONICA IN BOX                11785
    GIRLS ALPHABET IRON ON PATCHES       11235
    Name: Quantity, dtype: int64
    
    --- 2010-04 Cohort | RFM Level: Low ---
    Description
    CHINESE DRAGON PAPER LANTERNS     624
    LETTER SHAPE PENCIL SHARPENER      80
    PACK OF 60 DINOSAUR CAKE CASES     52
    Name: Quantity, dtype: int64
    
    --- 2010-04 Cohort | RFM Level: Medium ---
    Description
    FLAG OF ST GEORGE CAR FLAG    10201
    SOMBRERO                       2176
    ASSORTED COLOURS SILK FAN      2160
    Name: Quantity, dtype: int64
    
    --- 2010-04 Cohort | RFM Level: High ---
    Description
    ASSORTED COLOUR BIRD ORNAMENT         1758
    PACK OF 60 PINK PAISLEY CAKE CASES    1393
    JUMBO BAG RED RETROSPOT               1392
    Name: Quantity, dtype: int64
    
    --- 2010-05 Cohort | RFM Level: Low ---
    Description
    LARGE HANGING GLASS+ZINC LANTERN     560
    PAPER POCKET TRAVELING FAN           144
    SMALL STRIPES CHOCOLATE GIFT BAG      96
    Name: Quantity, dtype: int64
    
    --- 2010-05 Cohort | RFM Level: Medium ---
    Description
    PERIWINKLE T-LIGHT HOLDER         2972
    SET OF 4 NEW ENGLAND PLACEMATS    2664
    VINTAGE BEAD COSMETIC BAG         1728
    Name: Quantity, dtype: int64
    
    --- 2010-05 Cohort | RFM Level: High ---
    Description
    PACK OF 60 PINK PAISLEY CAKE CASES    1997
    60 TEATIME FAIRY CAKE CASES           1565
    PACK OF 72 SKULL CAKE CASES           1368
    Name: Quantity, dtype: int64
    
    --- 2010-06 Cohort | RFM Level: Low ---
    Description
    FENG SHUI PILLAR CANDLE                576
    COLOUR GLASS T-LIGHT HOLDER HANGING     80
    VICTORIAN GLASS HANGING T-LIGHT         66
    Name: Quantity, dtype: int64
    
    --- 2010-06 Cohort | RFM Level: Medium ---
    Description
    LUNCH BAG WOODLAND         1701
    LUNCH BAG CARS BLUE        1651
    LUNCH BAG  BLACK SKULL.    1615
    Name: Quantity, dtype: int64
    
    --- 2010-06 Cohort | RFM Level: High ---
    Description
    PACK OF 12 LONDON TISSUES          7806
    MINI PAINT SET VINTAGE             5976
    GIRLS ALPHABET IRON ON PATCHES     5760
    Name: Quantity, dtype: int64
    
    --- 2010-07 Cohort | RFM Level: Low ---
    Description
    BROCADE RING PURSE                  144
    DINOSAUR KEYRINGS ASSORTED          108
    12 PENCILS SMALL TUBE RED SPOTTY     96
    Name: Quantity, dtype: int64
    
    --- 2010-07 Cohort | RFM Level: Medium ---
    Description
    ASSORTED COLOURS SILK FAN      540
    SMALL CHINESE STYLE SCISSOR    500
    60 TEATIME FAIRY CAKE CASES    454
    Name: Quantity, dtype: int64
    
    --- 2010-07 Cohort | RFM Level: High ---
    Description
    PLACE SETTING WHITE HEART          2761
    ANTIQUE SILVER TEA GLASS ETCHED     807
    HEART OF WICKER SMALL               773
    Name: Quantity, dtype: int64
    
    --- 2010-08 Cohort | RFM Level: Low ---
    Description
    PINK  HONEYCOMB PAPER BALL       48
    ASSTD DESIGN BUBBLE GUM RING     30
    12 PENCIL SMALL TUBE WOODLAND    24
    Name: Quantity, dtype: int64
    
    --- 2010-08 Cohort | RFM Level: Medium ---
    Description
    SMALL POPCORN HOLDER                 4608
    WORLD WAR 2 GLIDERS ASSTD DESIGNS     721
    PACK OF 12 LONDON TISSUES             657
    Name: Quantity, dtype: int64
    
    --- 2010-08 Cohort | RFM Level: High ---
    Description
    PAPER CHAIN KIT VINTAGE CHRISTMAS    1053
    VINTAGE SNAP CARDS                   1001
    POPCORN HOLDER                        807
    Name: Quantity, dtype: int64
    
    --- 2010-09 Cohort | RFM Level: Low ---
    Description
    CHARLOTTE BAG SUKI DESIGN       100
    PACK OF 12 SUKI TISSUES          60
    ASSTD DESIGN BUBBLE GUM RING     60
    Name: Quantity, dtype: int64
    
    --- 2010-09 Cohort | RFM Level: Medium ---
    Description
    SET/6 FRUIT SALAD PAPER CUPS       7128
    SET/6 FRUIT SALAD  PAPER PLATES    7008
    POP ART PEN CASE & PENS            5184
    Name: Quantity, dtype: int64
    
    --- 2010-09 Cohort | RFM Level: High ---
    Description
    HEART OF WICKER SMALL                 6356
    HEART OF WICKER LARGE                 5253
    WHITE HANGING HEART T-LIGHT HOLDER    4632
    Name: Quantity, dtype: int64
    
    --- 2010-10 Cohort | RFM Level: Low ---
    Description
    SMALL CHINESE STYLE SCISSOR            100
    GRAND CHOCOLATECANDLE                   72
    SET OF 20 VINTAGE CHRISTMAS NAPKINS     52
    Name: Quantity, dtype: int64
    
    --- 2010-10 Cohort | RFM Level: Medium ---
    Description
    PAPER CHAIN KIT 50'S CHRISTMAS     798
    KEY FOB , SHED                     750
    KEY FOB , BACK DOOR                715
    Name: Quantity, dtype: int64
    
    --- 2010-10 Cohort | RFM Level: High ---
    Description
    RABBIT NIGHT LIGHT               3738
    ASSORTED COLOUR BIRD ORNAMENT     961
    PACK OF 12 COLOURED PENCILS       949
    Name: Quantity, dtype: int64
    
    --- 2010-11 Cohort | RFM Level: Low ---
    Description
    WRAP CHRISTMAS VILLAGE          125
    T-LIGHT HOLDER WHITE LACE        96
    LARGE CIRCULAR MIRROR MOBILE     96
    Name: Quantity, dtype: int64
    
    --- 2010-11 Cohort | RFM Level: Medium ---
    Description
    WORLD WAR 2 GLIDERS ASSTD DESIGNS    1059
    PARTY BUNTING                         806
    ASSORTED COLOUR BIRD ORNAMENT         714
    Name: Quantity, dtype: int64
    
    --- 2010-11 Cohort | RFM Level: High ---
    Description
    PACK OF 12 LONDON TISSUES       8451
    T-LIGHT GLASS FLUTED ANTIQUE    1698
    PLACE SETTING WHITE HEART       1626
    Name: Quantity, dtype: int64
    
    --- 2010-12 Cohort | RFM Level: Low ---
    Description
    RATTLE SNAKE EGGS                  192
    ANTIQUE SILVER TEA GLASS ETCHED    150
    POLKADOT RAIN HAT                   48
    Name: Quantity, dtype: int64
    
    --- 2010-12 Cohort | RFM Level: Medium ---
    Description
    ASSORTED FLOWER COLOUR "LEIS"         960
    AFGHAN SLIPPER SOCK PAIR              400
    VINTAGE HEADS AND TAILS CARD GAME     337
    Name: Quantity, dtype: int64
    
    --- 2010-12 Cohort | RFM Level: High ---
    Description
    PACK OF 12 TRADITIONAL CRAYONS     404
    CHARLOTTE BAG DOLLY GIRL DESIGN    398
    BASKET OF TOADSTOOLS               372
    Name: Quantity, dtype: int64
    
    --- 2011-01 Cohort | RFM Level: Low ---
    Description
    JUMBO BAG OWLS                     100
    ANTIQUE SILVER TEA GLASS ETCHED     72
    12 PENCILS SMALL TUBE SKULL         24
    Name: Quantity, dtype: int64
    
    --- 2011-01 Cohort | RFM Level: Medium ---
    Description
    FAIRY CAKE FLANNEL ASSORTED COLOUR    3123
    TEA TIME TEA TOWELS                   2600
    GIN + TONIC DIET METAL SIGN           2007
    Name: Quantity, dtype: int64
    
    --- 2011-01 Cohort | RFM Level: High ---
    Description
    VICTORIAN GLASS HANGING T-LIGHT    1650
    HANGING JAM JAR T-LIGHT HOLDER      672
    JUMBO BAG VINTAGE LEAF              525
    Name: Quantity, dtype: int64
    
    --- 2011-02 Cohort | RFM Level: Low ---
    Description
    AGED GLASS SILVER T-LIGHT HOLDER      144
    ORIGAMI VANILLA INCENSE CONES          96
    HEAVENS SCENT FRAGRANCE OILS ASSTD     72
    Name: Quantity, dtype: int64
    
    --- 2011-02 Cohort | RFM Level: Medium ---
    Description
    MINI PAINT SET VINTAGE           793
    MAGIC DRAWING SLATE SPACEBOY     631
    MAGIC DRAWING SLATE PURDEY       626
    Name: Quantity, dtype: int64
    
    --- 2011-02 Cohort | RFM Level: High ---
    Description
    WORLD WAR 2 GLIDERS ASSTD DESIGNS    10080
    RED  HARMONICA IN BOX                 8120
    BALLOON WATER BOMB PACK OF 35         3600
    Name: Quantity, dtype: int64
    
    --- 2011-03 Cohort | RFM Level: Low ---
    Description
    FELT EGG COSY CHICKEN              72
    VICTORIAN GLASS HANGING T-LIGHT    48
    FENG SHUI PILLAR CANDLE            48
    Name: Quantity, dtype: int64
    
    --- 2011-03 Cohort | RFM Level: Medium ---
    Description
    WORLD WAR 2 GLIDERS ASSTD DESIGNS    960
    PAPER CHAIN KIT EMPIRE               599
    DISCO BALL CHRISTMAS DECORATION      576
    Name: Quantity, dtype: int64
    
    --- 2011-03 Cohort | RFM Level: High ---
    Description
    WORLD WAR 2 GLIDERS ASSTD DESIGNS     6768
    RAIN PONCHO RETROSPOT                 4660
    PACK OF 60 PINK PAISLEY CAKE CASES    4561
    Name: Quantity, dtype: int64
    
    --- 2011-04 Cohort | RFM Level: Low ---
    Description
    LARGE CHINESE STYLE SCISSOR     100
    WRAP I LOVE LONDON               50
    ANTIQUE SILVER T-LIGHT GLASS     48
    Name: Quantity, dtype: int64
    
    --- 2011-04 Cohort | RFM Level: Medium ---
    Description
    ASSORTED COLOUR BIRD ORNAMENT         564
    PAPER CHAIN KIT EMPIRE                368
    SET OF 72 PINK HEART PAPER DOILIES    350
    Name: Quantity, dtype: int64
    
    --- 2011-04 Cohort | RFM Level: High ---
    Description
    BAG 125g SWIRLY MARBLES         612
    PARTY BUNTING                   441
    TRAVEL CARD WALLET KEEP CALM    384
    Name: Quantity, dtype: int64
    
    --- 2011-05 Cohort | RFM Level: Low ---
    Description
    JUMBO  BAG BAROQUE BLACK WHITE     100
    ANTIQUE SILVER T-LIGHT GLASS        72
    VICTORIAN GLASS HANGING T-LIGHT     48
    Name: Quantity, dtype: int64
    
    --- 2011-05 Cohort | RFM Level: Medium ---
    Description
    SMALL CERAMIC TOP STORAGE JAR     1374
    RETRO COFFEE MUGS ASSORTED         504
    TRADITIONAL MODELLING CLAY         374
    Name: Quantity, dtype: int64
    
    --- 2011-05 Cohort | RFM Level: High ---
    Description
    ANTIQUE SILVER T-LIGHT GLASS       1856
    VICTORIAN GLASS HANGING T-LIGHT    1553
    ZINC T-LIGHT HOLDER STARS SMALL     385
    Name: Quantity, dtype: int64
    
    --- 2011-06 Cohort | RFM Level: Low ---
    Description
    DECORATION SITTING BUNNY      96
    DINOSAUR KEYRINGS ASSORTED    72
    PLACE SETTING WHITE HEART     72
    Name: Quantity, dtype: int64
    
    --- 2011-06 Cohort | RFM Level: Medium ---
    Description
    BAG 125g SWIRLY MARBLES               2916
    VINTAGE SNAP CARDS                    1266
    GROW A FLYTRAP OR SUNFLOWER IN TIN    1200
    Name: Quantity, dtype: int64
    
    --- 2011-06 Cohort | RFM Level: High ---
    Description
    RABBIT NIGHT LIGHT                  615
    TRAVEL CARD WALLET I LOVE LONDON    528
    TRAVEL CARD WALLET KEEP CALM        499
    Name: Quantity, dtype: int64
    
    --- 2011-07 Cohort | RFM Level: Low ---
    Description
    BOTANICAL ROSE GREETING CARD           48
    COLUMBIAN CANDLE ROUND                 47
    CRYSTAL STUD EARRINGS ASSORTED COL     36
    Name: Quantity, dtype: int64
    
    --- 2011-07 Cohort | RFM Level: Medium ---
    Description
    GIRLS ALPHABET IRON ON PATCHES        1440
    GARDENERS KNEELING PAD CUP OF TEA      788
    WORLD WAR 2 GLIDERS ASSTD DESIGNS      624
    Name: Quantity, dtype: int64
    
    --- 2011-07 Cohort | RFM Level: High ---
    Description
    POPART WOODEN PENCILS ASST    500
    WRAP VINTAGE LEAF DESIGN      375
    WRAP POPPIES  DESIGN          250
    Name: Quantity, dtype: int64
    
    --- 2011-08 Cohort | RFM Level: Low ---
    Description
    RABBIT NIGHT LIGHT                  48
    BROCADE RING PURSE                  36
    PARTY CONE CHRISTMAS DECORATION     36
    Name: Quantity, dtype: int64
    
    --- 2011-08 Cohort | RFM Level: Medium ---
    Description
    STAR WOODEN CHRISTMAS DECORATION    618
    CHRISTMAS RETROSPOT ANGEL WOOD      588
    JAZZ HEARTS ADDRESS BOOK            440
    Name: Quantity, dtype: int64
    
    --- 2011-08 Cohort | RFM Level: High ---
    Description
    RABBIT NIGHT LIGHT                 369
    VICTORIAN GLASS HANGING T-LIGHT    292
    LIPSTICK PEN RED                   288
    Name: Quantity, dtype: int64
    
    --- 2011-09 Cohort | RFM Level: Low ---
    Description
    RAIN PONCHO RETROSPOT                 72
    WORLD WAR 2 GLIDERS ASSTD DESIGNS     48
    WOODEN TREE CHRISTMAS SCANDINAVIAN    48
    Name: Quantity, dtype: int64
    
    --- 2011-09 Cohort | RFM Level: Medium ---
    Description
    POPART WOODEN PENCILS ASST           700
    WORLD WAR 2 GLIDERS ASSTD DESIGNS    624
    ASSORTED COLOUR BIRD ORNAMENT        621
    Name: Quantity, dtype: int64
    
    --- 2011-09 Cohort | RFM Level: High ---
    Description
    WRAP CHRISTMAS VILLAGE               475
    CHRISTMAS HANGING STAR WITH BELL     392
    GARDENERS KNEELING PAD KEEP CALM     375
    Name: Quantity, dtype: int64
    
    --- 2011-10 Cohort | RFM Level: Low ---
    Description
    PACK OF 12 BLUE PAISLEY TISSUES        144
    WOODEN HEART CHRISTMAS SCANDINAVIAN    130
    CHRISTMAS TREE PAINTED ZINC            111
    Name: Quantity, dtype: int64
    
    --- 2011-10 Cohort | RFM Level: Medium ---
    Description
    POPCORN HOLDER                      634
    ASSORTED COLOURS SILK FAN           600
    CHRISTMAS HANGING STAR WITH BELL    538
    Name: Quantity, dtype: int64
    
    --- 2011-10 Cohort | RFM Level: High ---
    Description
    CHRISTMAS PUDDING TRINKET POT       1590
    CHRISTMAS DECOUPAGE CANDLE           703
    EMPIRE UNION JACK TV DINNER TRAY     601
    Name: Quantity, dtype: int64
    
    --- 2011-11 Cohort | RFM Level: Low ---
    Description
    ASSTD DESIGN 3D PAPER STICKERS         12540
    CHRISTMAS METAL POSTCARD WITH BELLS       72
    ASSORTED COLOURS SILK FAN                 60
    Name: Quantity, dtype: int64
    
    --- 2011-11 Cohort | RFM Level: Medium ---
    Description
    WOODEN STAR CHRISTMAS SCANDINAVIAN    321
    ROLL WRAP VINTAGE CHRISTMAS           288
    PENS ASSORTED SPACEBALL               288
    Name: Quantity, dtype: int64
    
    --- 2011-11 Cohort | RFM Level: High ---
    Description
    VINTAGE DOILY JUMBO BAG RED        1510
    GIRLS ALPHABET IRON ON PATCHES      576
    SPOTTY BUNTING                      320
    Name: Quantity, dtype: int64
    
    --- 2011-12 Cohort | RFM Level: Low ---
    Series([], Name: Quantity, dtype: int64)
    
    --- 2011-12 Cohort | RFM Level: Medium ---
    Description
    HAIRCLIPS FORTIES FABRIC ASSORTED    132
    BROCADE RING PURSE                    72
    SLEEPING CAT ERASERS                  61
    Name: Quantity, dtype: int64
    
    --- 2011-12 Cohort | RFM Level: High ---
    Description
    METAL SIGN TAKE IT OR LEAVE IT     1404
    HAND OVER THE CHOCOLATE   SIGN      632
    NATURAL SLATE HEART CHALKBOARD      621
    Name: Quantity, dtype: int64
    

## 코호트 별 고객 충성도(재구매 지속성)


```python
cohort_retention = dummy_df.groupby(['Customer ID', 'CohortMonth'])['PurchaseMonth'].nunique().reset_index()
cohort_retention.rename(columns={'PurchaseMonth': 'NumPurchaseMonths'}, inplace=True)

# CohortMonth별 평균 구매월 수
retention_summary = cohort_retention.groupby('CohortMonth')['NumPurchaseMonths'].mean().reset_index()

# 시각화
plt.figure(figsize=(10,5))
sns.barplot(data=retention_summary, x='CohortMonth', y='NumPurchaseMonths', palette='Blues_d')
plt.title('CohortMonth Average number of purchase months(loyalty)')
plt.xlabel('CohortMonth')
plt.ylabel('Average number of purchase months')
plt.yticks(range(1,11,1))
plt.xticks(rotation=45)
plt.show()
```

    C:\Users\Administrator\AppData\Local\Temp\ipykernel_4024\3696124218.py:9: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
    


    
![png](/CohortRetailshop/output_71_1.png)
    


## 제안 및 기대효과

>이번 분석을 통해 각 코호트 집단의 구매 성향과 충성도 패턴을 파악할 수 있었다.

>특히, 고객들은 가입 후 약 3개월까지 충성도가 상대적으로 높은 경향을 보이는 것을 확인하였다.

> 이를 바탕으로 다음과 같은 마케팅 전략을 제안을 해보겠다.

- 코호트 집단(2010-06,2010-09)를 대상으로 선호 제품에 대한 할인 쿠폰, 제품 품질 관련 광고나 이벤트와 같은 켐페인을 진행한다. (위에서 모두 뽑았듯이 다른 코호트 집단도 가능할 것이다.)
  > 단지, 2010-06과 2010-09는 직접적으로 선호 물품군이 다르고 구매 수량 등을 검정을 통해 찾아 냈기 때문에 특정한 것이다.

- 특히 3개월 차 시점에서 충성도가 높은 고객군이 존재하므로, 3개월 시점 전후에 타겟팅된 프로모션(쿠폰 제공, 리마인드 이메일 등) 을 집중적으로 운영하는 것이 효과적일 것으로 판단된다.
- 기대효과


>할인 쿠폰 제공 시 고객의 재구매 유도 가능성 증가

>제품 품질 강조를 통한 고객 신뢰도 및 브랜드 충성도 향상

>평균 구매 기간 연장 및 LTV 증대 기대

- 미래에 가능하다면?
  
향후 실제 캠페인 실행 후, 실험군 / 대조군 설정(A/B Test) 를 통해 해당 전략의 실효성을 검증하고, 예측 모델(리텐션/재구매율 예측 모델 등)을 구축하여 실무에 적용 가능할 것이다.


