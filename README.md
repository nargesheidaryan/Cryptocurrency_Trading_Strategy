SOL/EUR Trading Strategy Analysis — Profit Targets, Waiting Times & Drawdown Risk

This repository analyzes the feasibility and risk profile of a rule-based trading strategy on the SOL/EUR market.
The strategy uses fixed profit targets (3%, 10%, 15%) and evaluates how long it takes to reach them, as well as the maximum loss (drawdown) experienced before hitting each target.

This project is part of a larger analysis investigating whether daily trading with a 3% profit target is realistic under real market conditions.

📌 Project Goals

Evaluate whether a daily 3% profit target is achievable.

Determine the waiting time required for 3%, 10%, and 15% gains.

Measure maximum drawdown (worst loss before profit) for each target.

Understand the risk–reward trade-off of fixed-percentage trading strategies.

Identify whether longer holding periods produce more reliable outcomes.

📊 Dataset

Source:
Binance Historical Data — 1-Minute Candles (SOL/EUR)
https://data.binance.vision/?prefix=data/spot/monthly/klines/SOLEUR/1m/

Period used: January 2024 → July 2025

Each file contains 1-minute kline data:

open_time, open, high, low, close, volume,
close_time, quote_volume, trades,
taker_base_vol, taker_quote_vol, ignore

🛠 Data Processing Pipeline
✔ 1. Merge all monthly CSV files

Merges more than a year of 1-minute data without loading them into RAM, using streamed append mode.

✔ 2. Normalize timestamps

The open_time column may arrive in seconds / ms / µs depending on the source.
A custom function standardizes timestamps automatically.

✔ 3. Convert to datetime

Creates:

datetime (UTC timestamp)

date

time

✔ 4. Calculate profit targets

A Numba-JIT optimized loop searches forward in time for when price reaches:

+3%

+10%

+15%

Each target is processed in a separate Python script, including the one shown here (for +15%).

✔ 5. Compute waiting time

For each buy index:

duration = target_datetime - buy_datetime
duration_days = duration in days

✔ 6. Compute maximum drawdown before target

Using:

Sorted timestamps

A segment tree for fast minimum-price queries

A bisect search for time windows

Max loss formula:

loss = (buy_price – min_price_before_target) / buy_price * 100

✔ 7. Visualization & Probability Calculations

Plots:

Histogram of max losses

Probability distribution of loss thresholds

Wait-time distributions

📈 Example Script Included (for +15% Profit Target)

This repository contains separate Python scripts for:

3% profit

10% profit

15% profit (code shown in this project)

Each script calculates:

Time to reach the profit target

Maximum loss before profit

Histogram and probability of drawdowns

🔍 Key Findings (Final Report Summary)
1. 3% Daily Profit Is Not Realistic

Historical data shows that 3% profit typically requires ~2 days, not 1 day.

Daily trading with fixed thresholds is unreliable.

2. 10% Profit Target

Reached with ~70% probability after ~1 month.

3. 15% Profit Target

Requires ~39+ days on average.

High risk: Before hitting +15%, the price has a ~70% chance of falling ~21%.

4. Risk–Reward Balance

Forcing daily trades causes high risk and low success rates.

Longer holding periods make 10–15% profit more reasonable.

5. Practical Conclusion

A 3% target is possible, but not within 1 day.

A more realistic approach is minimum 2-day holding for small profit targets.

Larger profits are achievable only with long wait times and higher volatility risk.
