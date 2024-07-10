# Exploring Time Series Data With Python
### Kezhou Ren
### Presented at G11 Research Group, June 23, 2023
### Orgnized on July 9，2024
Time series data have certain characteristics that cross-sectional data do not, and these can require special attention when applying traditional estimators. In this lesson, we will investigate techniques for **Exploratory Analysis of Time Series Data** using an econometrics approach. 

We will utilize several Python packages to model time dependence and explore the characteristics of example data: *Statsmodels*, *Matplotlib*, *Seaborn*, and *Pandas Plotting*. The code examples in this notebook utilize each of these packages.

For this lesson we will be working with two time series datasets.The first dataset describes [the GDP fluctuations of America and Australia across 126 periods](https://data.worldbank.org/indicator/NY.GDP.MKTP.CD?end=2009&locations=AU-XU&start=1960), while the second dataset depicts [the minimum daily temperature over a span of 10 years in Melbourne](https://www.kaggle.com/datasets/samfaraday/daily-minimum-temperatures-in-me). **The ultimate goal is to understand the time dependency of target variables, enabling accurate predictions of their future trends.**

## Outline
**1. The nature of time series data**
 - **Preliminaries: CLRM**
 - **Properties of time series data** 
   - Time series data & Cross-sectional data
   - Impact Propensity & Long run propensity
   - Trend & Seasonality
   - Stationary & Non-stationary
 - **Visualizing time seires data with Python**
   - Line Plot
   - Hist Plot & Density Plot
   - Box Plot & Violin Plot
   - Heat Map
   - Lag Plot
   - Autocorrelation Plot 

**2. Models for time series analysis**
 - **Stationary univariate time series analysis** 
   - White Noise
   - Autoregressive Model (AR) & Moving Average (MA) Model
   - Autoregressive Moving Average (ARMA) Model
 - **Non-stationary univariate time series analysis**
   - Unit root test for stationary
   - Autoregressive Integration Moving Average (ARIMA) Model 
 - **Multi-variate time series analysis**
   - Autoregressive Distributed Lag (ARDL) Model
   - Vector Autoregressive (VAR) Model
   - Vector Error Corrected (VEC) Model

## Reference
[Wooldridge, Jeffrey M., 1960-. (2012). Introductory econometrics : a modern approach. Mason, Ohio :South-Western Cengage Learning](https://economics.ut.ac.ir/documents/3030266/14100645/Jeffrey_M._Wooldridge_Introductory_Econometrics_A_Modern_Approach__2012.pdf)

## Resources
In this tutorial, we will work with four powerful Python packages: **Pandas**, **Matplotlib**, **Statsmodels** and **Seaborn**. All four packages have extensive online documentation, we have provided the related resources for learning below.
- [**Statsmodels**](https://www.statsmodels.org/stable/api.htm)
- [**Visualization with Pandas**](https://pandas.pydata.org/pandas-docs/version/0.18.0/visualization.html)
- [**Matplotlib-Visual with Python**](https://matplotlib.org/stable/gallery/index.html)
- [**Seaborn: statistical data visualization**](https://seaborn.pydata.org/)

## About this Jupyter Notebook
This note book contains material to help you learn how to unravel data structure when analyzing time series. This notebook and the dataset can be downloaded from Github:

**https://github.com/XiWen0627/ExploringTimeSeriesDataWithPython**

The material are in the [/data](https://github.com/XiWen0627/ExploringTimeSeriesDataWithPython/tree/main/data) dictionary.

This notebook was constructed using Anaconda 3.8 Python distribution. If you are not running version Anaconda 3.5 or higher, we suggest you update your Anaconda distribution now. You can download the Python 3 Anaconda distribution for your operating system from the corresponding web site.
