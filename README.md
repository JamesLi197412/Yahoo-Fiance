## Stock Analysis (time series analysis)

### Introduction

In this repo, I would love to have practical experience on time-series analysis. One of the common example in time is Financial analysis, specfically stock analysis. exploration.py contains everything analyst could do on a time series dataset. 


In addition, Time series models utlised to predict future stock value, because they could recognize the trend and seasonality from the existing observatiosn (the past information) and then forecast a value based on its trend and seasonlity.


### Diagram

```mermaid
flowchart LR
    id1(Call API to gather stock intel) --> id2[Data Exploration/Visualisation] 
    id2[Data Exploration/Visualisation]  -- > id3[time series analysis]  
    id3[time series analysis]  -- > id4[Prediction]  

```

## Getting Started

### Prerequisites

```
    Tools Required:
    Visual Studio or Pycharm (Any IDE could run Python)
```

### Installing

A few libraries needed to install to ensure that the code could run.

Say what the step will be

```
    pip install yfiance
    pip install torch
    pip install statsmodels
```
1. Clone the repository
```
 git clone https://github.com/JamesLi197412/Yahoo-Fiance.git
```

### Forecast Model
##### ARIMA
 ARIMA models are effective for capturing linear trends and seasonal patterns.

#### LSTM
LSTM models tends to perform well on time series data with complex patterns and long-term dependencies.



## Authors

* **James Li** 
