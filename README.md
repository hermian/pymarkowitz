
# Pymarkowitz

<p align="left">
    <a href="https://www.python.org/">
        <img src="https://ForTheBadge.com/images/badges/made-with-python.svg"
            alt="python"></a> &nbsp;
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square"
            alt="MIT license"></a> &nbsp;
</p>

<!-- **Pymarkowitz** is an open source library for implementing portfolio optimisation. This library extends beyond the classical mean-variance optimization and takes into account a variety of risk and reward metrics, as well as the skew/kurtosis of assets. -->
Pymarkowitz는 포트폴리오 최적화를 구현하기위한 오픈 소스 라이브러리입니다. 이 라이브러리는 전통적인 평균 분산 최적화(mean-variance optimization)를 넘어서 확장되며 다양한 위험 및 보상 메트릭과 자산의 왜곡(skew)/첨도(kurtosis)를 고려합니다.

<!-- **Pymarkowitz** can aid your decision-making in portfolio allocation in a risk-efficient manner. Pymarkowitz covers major objectives and constraints related with major types of risk and reward metrics, as well as simulation to examine the relationship between all these metrics. The flexibility in its implementation gives you the maximum discretion to customize and suit it to your own needs.  -->
Pymarkowitz는 위험 효율적인(risk-efficient) 방식으로 포트폴리오 할당에 대한 의사 결정을 지원할 수 있습니다. Pymarkowitz는 주요 유형의 위험 및 보상 메트릭과 관련된 주요 목표 및 제약 사항을 다루며 이러한 모든 메트릭 간의 관계를 조사하기 위한 시뮬레이션을 다룹니다. 구현의 유연성은 사용자의 필요에 맞게 사용자 정의할 수있는 최대한의 재량을 제공합니다.

<!-- *Disclaimer: This library is for educational and entertainment purpose only. Please invest with due diligence at your own risk. -->
\* 면책 조항 :이 라이브러리는 교육 및 엔터테인먼트 목적으로만 사용됩니다. 귀하의 책임하에 실사를 통해 투자하십시오.

<!-- Head over to the directory **demos** to get an in-depth look at the project and its functionalities, or continue below to check out some brief examples. -->
디렉토리 demos로 이동하여 프로젝트와 그 기능을 자세히 살펴 보거나 아래에서 계속해서 몇 가지 간단한 예제를 확인하십시오.

---

## Table of Contents


- [Installation](#installation)
- [Features](#features)
- [Reference](#reference)
- [License](#license)

---

## Installation

### Setup

> install directly using pip

```shell
$ pip install pymarkowitz
```

> install from github

```shell
$ pip install git+https://github.com/johnsoong216/pymarkowitz.git
```

### Development

> For development purposes you can clone or fork the repo and hack right away!

```shell
$ git clone https://github.com/johnsoong216/pymarkowitz.git
```
---

## Features
- [Preprocessing](##preprocessing)
- [Optimization](##optimization)
- [Simulation](##simulation)
- [Backtesting](##backtesting)


---
### Preprocessing

<!-- > First step is to import all availble modules -->
> 1. 사용가능한 모든 모듈을 임포트한다

```python
import numpy as np
import pandas as pd
from pymarkowitz import *
```
<!-- > Read data with pandas. The dataset is available in the **datasets** directory. I will select 15 random stocks with 1000 observations -->
>Pandas로 데이터를 읽습니다. 데이터 세트는 datasets 디렉토리에서 사용할 수 있습니다. 1000 개의 관측치에서 15 개의 무작위 주식을 선택합니다.

```python
sp500 = pd.read_csv("datasets/sp500_1990_2000.csv", index_col='DATE').drop(["Unnamed: 0"], axis=1)
selected = sp500.iloc[:1000, np.random.choice(np.arange(0, sp500.shape[1]), 15, replace=False)]
```
<!-- > Use a ReturnGenerator to compute historical mean return and daily return. Note that there are a variety of options to compute rolling/continuous/discrete returns. Please refer to the **Return.ipynb** jupyter notebook in **demo** directory -->
>ReturnGenerator를 사용하여 과거 평균 수익률과 일일 수익률을 계산합니다. 롤링/연속/이산 수익률을 계산하는 다양한 옵션이 있습니다. demo 디렉토리의 **Return.ipynb** 주피터 노트북을 참조하십시오.

```python
ret_generator = ReturnGenerator(selected)
mu_return = ret_generator.calc_mean_return(method='geometric')
daily_return = ret_generator.calc_return(method='daily')
```

<!-- > Use a MomentGenerator to compute covariance/coskewness/cokurtosis matrix and beta. Note that there are a variety of options to compute the comoment matrix and asset beta, such as with semivariance, exponential and customized weighting. Normalizing matrices are also supported. Please refer to the **Moment(Covariance).ipynb** jupyter notebook in **demo** directory -->
> MomentGenerator를 사용하여 공분산(covariance)/공왜도(coskewness)/공첨도(cokurtosis) 행렬 및 베타를 계산합니다. 반분산, 지수 및 사용자 정의 가중치와 같이 공모멘트(comoment) 행렬 및 자산 베타를 계산하는 다양한 옵션이 있습니다. 정규화 행렬도 지원됩니다. demo 디렉토리의 **Moment(Covariance).ipynb** 주피터 노트북을 참조하십시오.

```python
benchmark = sp500.iloc[:1000].pct_change().dropna(how='any').sum(axis=1)/sp500.shape[1]
cov_matrix = mom_generator.calc_cov_mat()
beta_vec = mom_generator.calc_beta(benchmark)
```
<!-- > Construct higher moment matrices by calling -->
> 다음을 호출하여 더 높은 모멘트 행렬을 생성합니다.

```python
coskew_matrix = mom_generator.calc_coskew_mat()
cokurt_matrix = mom_generator.calc_cokurt_mat()
coseventh_matrix = mom_generator.calc_comoment_mat(7)
```

<!-- > Construct an Optimizer -->
>Optimizer를 구성합니다.

```python
PortOpt = Optimizer(mu_return, cov_matrix, beta_vec)
```

### Optimization

<!-- > Please refer to the **Optimization.ipynb** jupyter notebook in **demo** directory for more detailed explanations. -->
>자세한 설명은 **demo** 디렉토리의 **Optimization.ipynb** 주피터 노트북을 참조하세요.

<!-- > Set your Objective.  -->
>목표(objective)를 설정하십시오.

```python

### Call PortOpt.objective_options() to look at all available objectives

PortOpt.add_objective("min_volatility")

```

<!-- > Set your Constraints.  -->
>제약(constraint)을 설정하십시오.

```python
### Call PortOpt.constraint_options() to look at all available constraints.

PortOpt.add_constraint("weight", weight_bound=(-1,1), leverage=1) # Portfolio Long/Short
PortOpt.add_constraint("concentration", top_holdings=2, top_concentration=0.5) # Portfolio Concentration
```
<!-- > Solve and Check Summary -->
>해결(solve) 및 확인(check) 요약

```python
PortOpt.solve()
weight_dict, metric_dict = PortOpt.summary(risk_free=0.015, market_return=0.07, top_holdings=2)
```

```
# Metric Dict Sample Output
{'Expected Return': 0.085,
 'Leverage': 1.0001,
 'Number of Holdings': 5,
 'Top 2 Holdings Concentrations': 0.5779,
 'Volatility': 0.1253,
 'Portfolio Beta': 0.7574,
 'Sharpe Ratio': 0.5586,
 'Treynor Ratio': 0.0924,
 "Jenson's Alpha": 0.0283}
 
# Weight Dict Sample Output
{'GIS': 0.309, 'CINF': 0.0505, 'USB': 0.104, 'HES': 0.2676, 'AEP': 0.269}
```

### Simulation

<!-- > Simulate and Select the Return Format (Seaborn, Plotly, DataFrame). DataFrame Option will also have the random weights used in each iteration. -->
> 반환 형식 (Seaborn, Plotly, DataFrame)을 시뮬레이션하고 선택합니다. DataFrame 옵션에는 각 반복에서 사용되는 임의 가중치도 있습니다.

<!-- > Please refer to the **Simulation.ipynb** jupyter notebook in **demo** directory for more detailed explanations. -->
>자세한 설명은 **demo** 디렉토리의 **Simulation.ipynb** 주피터 노트북을 참조하십시오.

```python
### Call Portopt.metric_options to see all available options for x, y axis

PortOpt.simulate(x='expected_return', y='sharpe', y_var={"risk_free": 0.02}, iters=10000, weight_bound=(-1, 1), leverage=1, ret_format='sns')
```
![Sharpe VS Return](https://github.com/johnsoong216/pymarkowitz/blob/master/images/return_vs_sharpe.png)


### Backtesting

<!-- > Use **pymarkowitz** to construct optimized weights and backtest with real life portfolio.
In this example, I am using SPDR sector ETFs to construct an optimized portfolio and compare against buy & hold SPY. -->
> pymarkowitz를 사용하여 최적화 된 가중치를 구성하고 실제 포트폴리오로 백테스트하십시오. 이 예에서는 SPDR 섹터 ETF를 사용하여 최적화 된 포트폴리오를 구성하고 SPY 매수 및 보유와 비교합니다.

---

```python
import bt

data = bt.get('spy, rwr, xlb, xli, xly, xlp, xle, xlf, xlu, xlv, xlk', start='2005-01-01')
```

<!-- > The configurations can be adjusted flexibly, please check backtesting.ipynb in demo directory for more detail. In this case we are minimizing volatility with a capped weight of 25% on each sector. -->
>구성은 유연하게 조정할 수 있습니다. 자세한 내용은 demo 디렉토리의 backtesting.ipynb를 확인하십시오. 이 경우 각 섹터에 대해 25 %의 제한 가중치로 변동성을 최소화하고 있습니다.

```python
strategy = WeighMarkowitz(Config) #Imported from pymarkowitz.backtester.py

# Personal Strategy
s1 = bt.Strategy('s1', [bt.algos.RunWeekly(),
                       bt.algos.SelectAll(),
                       strategy,
                       bt.algos.Rebalance()])
test1 = bt.Backtest(s1, data)

# Buy & Hold
s2 = bt.Strategy('s2', [bt.algos.RunWeekly(),
                       bt.algos.SelectAll(),
                       bt.algos.WeighEqually(),
                       bt.algos.Rebalance()])
test2 = bt.Backtest(s2, data[['spy']].iloc[Config.lookback:])
res = bt.run(test1, test2)
res.plot()
```
![Backtest_Result](https://github.com/johnsoong216/pymarkowitz/blob/master/images/backtest_sector_vs_spy.PNG)


---

## Reference

Calculations of **Correlation, Diversifcation & Risk Parity Factors**:
<br>
https://investresolve.com/file/pdf/Portfolio-Optimization-Whitepaper.pdf

Calculations for **Sharpe, Sortino, Beta, Treynor, Jenson's Alpha**:
<br>
https://www.cfainstitute.org/-/media/documents/support/programs/investment-foundations/19-performance-evaluation.ashx?la=en&hash=F7FF3085AAFADE241B73403142AAE0BB1250B311
<br>
https://www.investopedia.com/terms/j/jensensmeasure.asp
<br>
https://www.investopedia.com/ask/answers/070615/what-formula-calculating-beta.asp
<br>

Calculations for **Higher Moment Matrices**:
<br>
https://cran.r-project.org/web/packages/PerformanceAnalytics/vignettes/EstimationComoments.pdf
<br>


---

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright 2020 ©
