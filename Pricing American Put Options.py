
                    ##### Pricing American Put Options: CRR Trees and Longstaff–Schwartz Monte Carlo #####

import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import datetime


spy = yf.Ticker("SPY")
expirations = spy.options
today = datetime.date.today()

# Shared Parameters
S = spy.history(period="1d")['Close'].iloc[-1]
q = (spy.info.get('dividendYield', 0) or 0) / 100
r = yf.Ticker('^IRX').history(period='1d')['Close'].iloc[-1] / 100

# Picking first expiration at least 21 days out
exp = next(e for e in expirations
           if (datetime.datetime.strptime(e, "%Y-%m-%d").date() - today).days >= 21)
expiry_date = datetime.datetime.strptime(exp, "%Y-%m-%d").date()
T = (expiry_date - today).days / 365

print(f"Expiry: {exp} | T = {T:.4f} years | S = {S:.2f} | r = {r:.4f} | q = {q:.4f}")

# Puts DataFrame
puts = spy.option_chain(exp).puts.copy()
puts['T'] = T
puts['IV'] = puts['impliedVolatility']
puts['Mid Price'] = (puts['bid'] + puts['ask']) / 2


# 1a) Plotting Implied Volatilities

plt.figure(figsize=(10, 5))
plt.plot(puts['strike'], puts['IV'], color='black', marker='o', linestyle='none', fillstyle='none')
plt.title(f"Implied Volatility vs Strike ({exp})")
plt.xlabel('Strike Price ($)')
plt.ylabel('Implied Volatility')
plt.tight_layout()
plt.show()


# 1b) Black-Scholes European Put Pricing (w/ continuous div yield))

def bs_put_price(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

puts['bs_price'] = puts.apply(
    lambda row: bs_put_price(S, row['strike'], row['T'], r, q, row['IV']), axis=1)

print("\n1b) Black-Scholes Prices:")
print(puts[['strike', 'Mid Price', 'IV', 'bs_price']].head(20))


# 1c) CRR Binomial Tree - American Put Pricing

def crr_american_put(S, K, T, r, q, sigma, N=200):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    ST = np.array([S * (u ** (N - 2*j)) for j in range(N + 1)])
    V = np.maximum(K - ST, 0)
    for i in range(N - 1, -1, -1):
        ST = ST[:-1] / u
        hold = discount * (p * V[:-1] + (1 - p) * V[1:])
        exercise = np.maximum(K - ST, 0)
        V = np.maximum(hold, exercise)
    return V[0]

puts['crr_price'] = puts.apply(
    lambda row: crr_american_put(S, row['strike'], row['T'], r, q, row['IV'], N=200), axis=1)

print("\n1c) CRR Binomial Tree Prices:")
print(puts[['strike', 'Mid Price', 'bs_price', 'crr_price']].head(20))


# 1d) Monte Carlo (Longstaff–Schwartz) - American Put Pricing

def lsm_american_put(S0, K, T, r, q, sigma, M=20000, N=50, seed=123):
    if T <= 0 or sigma <= 0:
        return max(K - S0, 0)

    rng = np.random.default_rng(seed)
    dt = T / N
    disc = np.exp(-r * dt)

    # Simulating GBM paths under risk-neutral measure
    Z = rng.standard_normal((M, N))
    increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    S = np.empty((M, N + 1))
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(np.cumsum(increments, axis=1))

    # Intrinsic values for a put
    H = np.maximum(K - S, 0.0)

    # Cashflows: start with maturity payoff
    CF = H[:, -1].copy()

    # Backward induction
    for t in range(N - 1, 0, -1):
        CF *= disc

        itm = H[:, t] > 0  # in-the-money paths at time t
        if np.any(itm):
            X = S[itm, t]

            # Estimate continuation value via regression on basis functions
            # (common choice: 1, X, X^2)
            A = np.column_stack([np.ones_like(X), X, X**2])
            beta, *_ = np.linalg.lstsq(A, CF[itm], rcond=None)
            C_hat = A @ beta

            exercise = H[itm, t] > C_hat
            idx_itm = np.where(itm)[0]

            # If exercised, continuation replaced w/ immediate payoff
            CF[idx_itm[exercise]] = H[idx_itm[exercise], t]

    # Discounting back to time 0
    price = np.mean(CF) * disc
    return price


puts['mc_price'] = puts.apply(
    lambda row: lsm_american_put(S, row['strike'], row['T'], r, q, row['IV'],
                                 M=20000, N=50, seed=123),
    axis=1
)

print("\n1d) LSM Monte Carlo Prices:")
print(puts[['strike', 'Mid Price', 'bs_price', 'crr_price', 'mc_price']].head(20))


# 2) RMSE Table

puts['bs_error']  = (puts['bs_price']  - puts['Mid Price'])**2
puts['crr_error'] = (puts['crr_price'] - puts['Mid Price'])**2
puts['mc_error']  = (puts['mc_price']  - puts['Mid Price'])**2

rmse = {
    'Black-Scholes RMSE': np.sqrt(puts['bs_error'].mean()),
    'CRR Binomial RMSE':  np.sqrt(puts['crr_error'].mean()),
    'Monte Carlo RMSE':   np.sqrt(puts['mc_error'].mean())
}

print("\n2) RMSE Comparison:")
for k, v in rmse.items():
    print(f"  {k}: {v:.6f}")

best = min(rmse, key=rmse.get)
print(f"\n  Most accurate method: {best}")


# 3) Vary steps (CRR) and simulations (LSM MC)

crr_steps = [50, 100, 200, 500, 1000]
mc_sims   = [100, 200, 500, 1000, 2000]

lsm_steps = 50

crr_rmse_list = []
mc_rmse_list  = []


# CRR: Varying N Steps
for N in crr_steps:
    puts[f'crr_{N}'] = puts.apply(
        lambda row: crr_american_put(S, row['strike'], row['T'], r, q, row['IV'], N=N),
        axis=1
    )
    rmse_val = np.sqrt(((puts[f'crr_{N}'] - puts['Mid Price'])**2).mean())
    crr_rmse_list.append(rmse_val)
    print(f"CRR N={N}: RMSE = {rmse_val:.6f}")

# LSM MC: Varying M Paths
for M in mc_sims:
    puts[f'mc_{M}'] = puts.apply(
        lambda row: lsm_american_put(
            S0=S,
            K=row['strike'],
            T=row['T'],
            r=r,
            q=q,
            sigma=row['IV'],
            M=M,          # number of paths
            N=lsm_steps,  # number of exercise dates
            seed=123      
        ),
        axis=1
    )
    rmse_val = np.sqrt(((puts[f'mc_{M}'] - puts['Mid Price'])**2).mean())
    mc_rmse_list.append(rmse_val)
    print(f"LSM MC M={M} (N={lsm_steps}): RMSE = {rmse_val:.6f}")


# Plotting RMSE vs steps/simulations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(crr_steps, crr_rmse_list, marker='o')
ax1.set_title('CRR RMSE vs Number of Steps')
ax1.set_xlabel('Number of Steps (N)')
ax1.set_ylabel('RMSE')

ax2.plot(mc_sims, mc_rmse_list, marker='o')
ax2.set_title(f'LSM Monte Carlo RMSE vs Number of Paths (N={lsm_steps})')
ax2.set_xlabel('Number of Paths (M)')
ax2.set_ylabel('RMSE')

plt.tight_layout()
plt.show()



