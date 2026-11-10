# Reference URL: https://www.youtube.com/watch?v=Isutk-wqJfE&t=728s
# Efficient Frontier Visualisation for Indonesian Stocks (IDX)

import numpy as np
import datetime as dt
import pandas as pd
from scipy import optimize as sc
import yfinance as yf
import plotly.graph_objects as go


# ==============================================================
# IMPORT DATA
# ==============================================================

def getData(stock, start, end):
    """Download stock data and compute log returns, mean returns, and covariance matrix."""
    stockData = yf.download(stock, start=start, end=end, auto_adjust=True)
    #         Select 'Close' level and rename columns to ticker symbols.
    if isinstance(stockData.columns, pd.MultiIndex):
        stockData = stockData['Close']
    else:
        stockData = pd.DataFrame(stockData['Close'])
        stockData.columns = stock

    returns = np.log(stockData / stockData.shift(1))
    mean_returns = returns.mean()
    covMatrix = returns.cov()

    return mean_returns, covMatrix


# ==============================================================
# PORTFOLIO PERFORMANCE
# ==============================================================

def portfolioPerformance(weights, meanReturns, covMatrix):
    """
    Returns annualised portfolio return and annualised portfolio volatility.

    FIX 4: The .ipynb version divided std by sqrt(252) instead of multiplying —
            that is WRONG. Annualised volatility = daily_vol * sqrt(252).
            Corrected formula: std * sqrt(252)
    """
    returns = np.sum(meanReturns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(252)
    return returns, std


# ==============================================================
# STOCK LIST
# ==============================================================

stocklist = ['ANTM', 'ADRO', 'PTBA', 'BBCA', 'BBRI', 'ASII']
stock = [s + '.JK' for s in stocklist]  # FIX 5: variable name shadowed 'stock' loop var → renamed to 's'

endDate   = dt.datetime.now()
startDate = endDate - dt.timedelta(days=5 * 365)

# Weight initialisation (equal-weight)
numAssets = len(stock)
weights   = np.array(numAssets * [1. / numAssets])

# Indonesia 10-year Government Bond yield (SBN)
RISK_FREE_RATE = 6.53 / 100

# Download data once at module level
meansReturns, covMatrix = getData(stock, start=startDate, end=endDate)
returns, std = portfolioPerformance(weights, meansReturns, covMatrix)

print("Equal-weight Portfolio  |  Return: {:.2f}%  |  Volatility: {:.2f}%".format(
    returns * 100, std * 100))


# ==============================================================
# OPTIMISATION HELPERS
# ==============================================================

def negativeSR(weights, meanReturns, covMatrix, riskFreeRate=RISK_FREE_RATE):
    """Negative Sharpe Ratio (to be minimised)."""
    pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix)
    return -(pReturns - riskFreeRate) / pStd


def maxSR(meanReturns, covMatrix, riskFreeRate=RISK_FREE_RATE, constraintSet=(0, 1)):
    """Find the portfolio weights that maximise the Sharpe Ratio."""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraintSet for _ in range(numAssets))
    result = sc.minimize(
        negativeSR, numAssets * [1. / numAssets],
        args=args, method='SLSQP',
        bounds=bounds, constraints=constraints
    )
    return result


def portfolioVariance(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[1]


def portfolioReturn(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[0]


def minimizeVariance(meanReturns, covMatrix, constraintSet=(0, 1)):
    """Find the minimum-variance portfolio."""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraintSet for _ in range(numAssets))
    result = sc.minimize(
        portfolioVariance, numAssets * [1. / numAssets],
        args=args, method='SLSQP',
        bounds=bounds, constraints=constraints
    )
    return result


def efficient_opt(meanReturns, covMatrix, returnTarget, constraintSet=(0, 1)):
    """For a given return target, minimise portfolio variance (Efficient Frontier point)."""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = (
        {'type': 'eq', 'fun': lambda x: portfolioReturn(x, meanReturns, covMatrix) - returnTarget},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    )
    bounds = tuple(constraintSet for _ in range(numAssets))
    effOpt = sc.minimize(
        portfolioVariance, numAssets * [1. / numAssets],
        args=args, method='SLSQP',
        bounds=bounds, constraints=constraints
    )
    return effOpt


# ==============================================================
# COMBINED ANALYSIS
# ==============================================================

def calculatedResults(meanReturns, covMatrix, riskFreeRate=RISK_FREE_RATE, constraintSet=(0, 1)):
    """
    Returns:
        maxSR_returns, maxSR_std, maxSR_allocation,
        minVol_returns, minVol_std, minVol_allocation,
        efficientList (list of volatilities), targetReturn (array of returns)

    FIX 6: Original code called calculatedResults with riskFreeRate=0 by default
            (inside EF_graph). Now consistently uses RISK_FREE_RATE.
    FIX 7: Variable name typo — 'minSR_std' should be 'minVol_std' for clarity.
    """
    # Max Sharpe Ratio portfolio
    maxSR_portfolio = maxSR(meanReturns, covMatrix, riskFreeRate)
    maxSR_returns, maxSR_std = portfolioPerformance(maxSR_portfolio['x'], meanReturns, covMatrix)

    maxSR_allocation = pd.DataFrame(
        maxSR_portfolio['x'], index=meanReturns.index, columns=['allocation']
    )
    maxSR_allocation['allocation'] = [round(i * 100, 2) for i in maxSR_allocation['allocation']]

    # Minimum Volatility portfolio
    minVol_portfolio = minimizeVariance(meanReturns, covMatrix, constraintSet)
    minVol_returns, minVol_std = portfolioPerformance(minVol_portfolio['x'], meanReturns, covMatrix)

    minVol_allocation = pd.DataFrame(
        minVol_portfolio['x'], index=meanReturns.index, columns=['allocation']
    )
    minVol_allocation['allocation'] = [round(i * 100, 2) for i in minVol_allocation['allocation']]

    # Efficient Frontier — sweep returns from minVol to maxSR
    efficientList = []
    # FIX 8: Extend frontier slightly beyond maxSR to show the full curve.
    #         Also handle edge case where minVol_returns >= maxSR_returns.
    low  = min(minVol_returns, maxSR_returns)
    high = max(minVol_returns, maxSR_returns)
    targetReturn = np.linspace(low, high, 50)   # 50 points → smoother curve

    for target in targetReturn:
        result = efficient_opt(meanReturns, covMatrix, target)
        if result.success:
            efficientList.append(result['fun'])
        else:
            efficientList.append(np.nan)

    # Convert to percentages for display
    maxSR_returns_pct = round(maxSR_returns * 100, 2)
    maxSR_std_pct     = round(maxSR_std     * 100, 2)
    minVol_returns_pct= round(minVol_returns* 100, 2)
    minVol_std_pct    = round(minVol_std    * 100, 2)

    return (
        maxSR_returns_pct, maxSR_std_pct,     maxSR_allocation,
        minVol_returns_pct, minVol_std_pct,   minVol_allocation,
        efficientList, targetReturn
    )


# ==============================================================
# MONTE CARLO SIMULATION
# ==============================================================

def monteCarloSim(meanReturns, covMatrix, numPortfolios=10000, riskFreeRate=RISK_FREE_RATE):
    """
    Simulate random portfolios to visualise the attainable set.

    FIX 9: Added riskFreeRate parameter (was using global RISK_FREE_RATE directly —
            works but coupling is bad practice).
    """
    numAssets = len(meanReturns)
    results   = np.zeros((3, numPortfolios))

    for i in range(numPortfolios):
        w = np.random.random(numAssets)
        w /= np.sum(w)
        ret, vol   = portfolioPerformance(w, meanReturns, covMatrix)
        sharpe     = (ret - riskFreeRate) / vol
        results[0, i] = ret * 100   # FIX 10: convert to % for consistent axis units
        results[1, i] = vol * 100
        results[2, i] = sharpe

    return results


# ==============================================================
# EFFICIENT FRONTIER PLOT
# ==============================================================

def EF_graph(meanReturns, covMatrix, riskFreeRate=RISK_FREE_RATE, constraintSet=(0, 1)):
    """
    Plot Efficient Frontier, Max SR point, Min Vol point, and Monte Carlo cloud.

    FIX 11: EF_curve used mode='markers' but specified line= style → changed to
             mode='lines' so the curve is drawn as a continuous line.
    FIX 12: Monte Carlo scatter added so each individual emiten's risk/return
             context is visible.
    FIX 13: Added individual stock scatter so emiten positions are labelled on the chart.
    """
    (maxSR_returns, maxSR_std,
     maxSR_allocation,
     minVol_returns, minVol_std,
     minVol_allocation,
     efficientList, targetReturn) = calculatedResults(meanReturns, covMatrix, riskFreeRate, constraintSet)

    # ── Monte Carlo cloud ──────────────────────────────────────
    mc = monteCarloSim(meanReturns, covMatrix, numPortfolios=5000, riskFreeRate=riskFreeRate)

    MC_scatter = go.Scatter(
        name='Monte Carlo Portfolios',
        mode='markers',
        x=mc[1],   # volatility %
        y=mc[0],   # return %
        marker=dict(
            color=mc[2],          # colour by Sharpe Ratio
            colorscale='Viridis',
            size=4,
            opacity=0.5,
            colorbar=dict(title='Sharpe Ratio', thickness=15)
        )
    )

    # ── Individual stocks ──────────────────────────────────────
    stock_returns_pct = []
    stock_std_pct     = []
    for i in range(len(meanReturns)):
        w_single = np.zeros(len(meanReturns))
        w_single[i] = 1.0
        r, s = portfolioPerformance(w_single, meanReturns, covMatrix)
        stock_returns_pct.append(round(r * 100, 2))
        stock_std_pct.append(round(s * 100, 2))

    Individual_stocks = go.Scatter(
        name='Individual Stocks',
        mode='markers+text',
        x=stock_std_pct,
        y=stock_returns_pct,
        text=list(meanReturns.index),
        textposition='top center',
        marker=dict(color='orange', size=10, symbol='diamond',
                    line=dict(width=1, color='black'))
    )

    # ── Efficient Frontier curve ───────────────────────────────
    ef_x = [round(ef_std * 100, 2) for ef_std in efficientList]
    ef_y = [round(target * 100, 2) for target in targetReturn]

    EF_curve = go.Scatter(
        name='Efficient Frontier',
        mode='lines',                       # ← FIXED (was 'markers')
        x=ef_x,
        y=ef_y,
        line=dict(color='black', width=4, dash='dashdot')
    )

    # ── Max Sharpe Ratio point ─────────────────────────────────
    MaxSharpeRatio = go.Scatter(
        name='Maximum Sharpe Ratio',
        mode='markers',
        x=[maxSR_std],
        y=[maxSR_returns],
        marker=dict(color='red', size=14, line=dict(width=3, color='black'))
    )

    # ── Min Volatility point ───────────────────────────────────
    MinVol = go.Scatter(
        name='Minimum Volatility',
        mode='markers',
        x=[minVol_std],
        y=[minVol_returns],
        marker=dict(color='green', size=14, line=dict(width=3, color='black'))
    )

    # ── Layout ─────────────────────────────────────────────────
    layout = go.Layout(
        title=dict(
            text=(
                f'<b>Efficient Frontier — Indonesian Stocks (IDX)</b><br>'
                f'<sup>Stocks: {", ".join(stocklist)}  |  '
                f'Risk-Free Rate: {riskFreeRate*100:.2f}% (SBN 10Y)</sup>'
            ),
            x=0.5
        ),
        yaxis=dict(title='Annualised Return (%)'),
        xaxis=dict(title='Annualised Volatility (%)'),
        showlegend=True,
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='#E2E2E2',
            bordercolor='black',
            borderwidth=2
        ),
        width=950,
        height=650
    )

    fig = go.Figure(data=[MC_scatter, Individual_stocks, EF_curve, MaxSharpeRatio, MinVol],
                    layout=layout)

    # ── Annotations for allocation ─────────────────────────────
    maxSR_text = '<br>'.join([
        f"{idx}: {row['allocation']}%"
        for idx, row in maxSR_allocation.iterrows()
    ])
    minVol_text = '<br>'.join([
        f"{idx}: {row['allocation']}%"
        for idx, row in minVol_allocation.iterrows()
    ])

    fig.add_annotation(
        x=maxSR_std, y=maxSR_returns,
        text=f"<b>Max SR</b><br>{maxSR_text}",
        showarrow=True, arrowhead=2,
        bgcolor='lightyellow', bordercolor='red',
        xshift=10, yshift=10
    )
    fig.add_annotation(
        x=minVol_std, y=minVol_returns,
        text=f"<b>Min Vol</b><br>{minVol_text}",
        showarrow=True, arrowhead=2,
        bgcolor='lightcyan', bordercolor='green',
        xshift=10, yshift=-40
    )

    fig.show()
    return fig


# ==============================================================
# MAIN
# ==============================================================

if __name__ == '__main__':
    print("\n--- Calculated Results ---")
    (mSR_ret, mSR_std, mSR_alloc,
     mV_ret,  mV_std,  mV_alloc,
     effList, tgtRet) = calculatedResults(meansReturns, covMatrix)

    print(f"\nMax Sharpe Ratio Portfolio  |  Return: {mSR_ret}%  |  Volatility: {mSR_std}%")
    print(mSR_alloc)
    print(f"\nMin Volatility Portfolio    |  Return: {mV_ret}%  |  Volatility: {mV_std}%")
    print(mV_alloc)

    print("\n--- Monte Carlo Simulation (first 5 rows) ---")
    mc_results = monteCarloSim(meansReturns, covMatrix, numPortfolios=10000)
    print(pd.DataFrame(mc_results.T, columns=['Return%', 'Volatility%', 'SharpeRatio']).head())

    print("\n--- Plotting Efficient Frontier ---")
    EF_graph(meansReturns, covMatrix)
