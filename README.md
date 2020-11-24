# Cryptocurrency Returns Predictor

## Project Description
### Introduction

* **Objective**: Application of Random Forests (project for my intern at
[Research Center VERA](https://www.unive.it/pag/35190/), Ca' Foscari University of Venice).   

* **Abstract**: Use sentiment-based features to predict cryptocurrency returns.
Models used: Random Forest Classifier, Random Forest Regressor, and VAR time-series model.
Analysis timeframe: 28/11/2014 - 25/07/2020.

* **Status**: Completed.

### Methods Used
* Random Forests (Regressor & Classifier).
* Principal Component Analysis.
* Vector Autoregression (VAR) model.
* Sentiment indicators (from my graduation thesis).

### Dependencies
* Python 3
* numpy==1.18.5
* pandas==1.0.5
* scikit-learn==0.23.2
* statsmodels==0.12.0
* plotly==4.9.0

### Interesting Results to Keep You Reading
Backtesting strategies based on 3 models:   
* How to generate trading signals: Long as predicted return > 0, short as predicted return < 0, wait otherwise.
* Test set (25% of the dataset): 05/03/2019 - 25/07/2020
* RF Classifier outperforms significantly both strategies and also the simple buy-and-hold strategy.
![alt text](https://github.com/dang-trung/crypto-return-predictor/blob/master/figures/strats.png)
* See the [interactive version](https://github.com/dang-trung/crypto-return-predictor/blob/master/figures/strats.html).

## Table of Contents
https://ecotrust-canada.github.io/markdown-toc/

## Getting Started

### How to Run
1. Clone this repo:  
`git clone https://github.com/dang-trung/crypto-return-predictor`
2. Create your environment (virtualenv):  
`virtualenv -p python3 venv`  
`source venv/bin/activate` (bash) or `venv\Scripts\activate` (windows)   
`(venv) cd crypto-return-predictor`  
`(venv) pip install -e`  

    Or (conda):  
`conda env create -f environment.yml`  
`conda activate crypto-return-predictor`  
3. Run in terminal:  
`python -m crypto_return_predictor`  

### Dependent Variable/Target
Cryptocurrency market returns (computed using the market index CRIX,
retrieved [here](http://data.thecrix.de/data/crix.json),
see more on how the index is created at [Trimborn & Härdle (2018)](https://doi.org/10.1016/j.jempfin.2018.08.004)
or [those authors' website](https://thecrix.de/).)

### Sentiment Measures
* Sentiment score of Messages on StockTwits, Reddit Submissions, Reddit Comments
  * Computed using dictionary-based sentiment analysis, lexicon used: crypto-specific lexicon by [Chen et al (2019)](http://dx.doi.org/10.2139/ssrn.3398423),
  retrieved at the main author's [personal page](https://sites.google.com/site/professorcathychen/resume).
  * StockTwits messages are retrieved through [StockTwits Public API](https://api.stocktwits.com/developers),
    Reddit data are retrieved using [PushShift.io Reddit API](https://github.com/pushshift/api).
* Messages volume on StockTwits, Reddit Submissions, Reddit Comments.
* Market volatility index VCRIX (see how the index is created: [Kolesnikova (2018)](https://edoc.hu-berlin.de/bitstream/handle/18452/20056/master_kolesnikova_alisa.pdf?sequence=3&isAllowed=y), retrieved [here](http://data.thecrix.de/data/crix11.json).)
* Market trading volume (retrieved using [Nomics Public API](https://docs.nomics.com/))

_Read more on how I retrieve these sentiment measures in my [graduation thesis](https://github.com/dang-trung/) or its Github [repo](https://github.com/dang-trung/)._

### Features Selection
* For VAR model: Lagged values of the first principal component of all 9 sentiment measures (up to 5 lags).
* For Random Forests: Sentiment measures' lagged Values (up to 5 lags).

## Results

## Read More
For better understanding of the project, kindly read the [report]([link-to-report]).
