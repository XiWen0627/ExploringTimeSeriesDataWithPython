# Chapter 1. The Nature of Time Series Data
In this chapter, we begin to delve into the properties of Ordinary Least Square (OLS) for estimating linear regression models using time series data.

## Before we take off, you need to know ···
### In general, data can be simply classified to four types:
  1. Cross-sectional data
  2. **Time series data**
  3. Panel data
  4. Pooled cross section

### Assumptions of Classical Linear Regression Models
- **MLR.1 Linear in Parameters**

  The model in the population can be written as
  $$ y=\beta_0+\beta_1x_1+\beta2x_2+···+\beta_kx_k+u $$
  where ${\beta_1, \beta2, ···, \beta_k}$ are the unknown parameters (constants) of interest and $u$ is an unobserved random error or disturbance term.  
- **MLR.2 Random Sampling**

- **MLR.3 No Perfect Collinearity**

  none of the independent variables is constant, and there are no exact linear relationships among the independent variables.
  
- **MLR.4 Zero Conditional Mean**
  
  The error $u$ has an expected value of zero given any values of the independent variables
  $$E(u|x_1, x_2, ···, x_k)=0$$
  
- **MLR.5 Homoskedasticity**

  The error $u$ has the same variance given any values of the explanatory variables
  $$Var(u|x_1, x_2, ···, x_k)=\sigma^2$$

- **MLR.6 Normality in Error Terms**

### Tips
- **Under assumptions MLR.1 - MLR.4** the OLS estimators are unbiased estimators of the population parameters.

- **Under assumptions MLR.1 - MLR.5** Gauss-Markov Theorem: Best Linear Unbiased Estimators (BLUEs).

- **Under assumptions MLR.1 - MLR.6** Assumptions of the Classical Linear Regression Model (CLRM).
