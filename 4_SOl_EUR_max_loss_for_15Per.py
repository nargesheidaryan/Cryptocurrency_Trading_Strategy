import pandas as pd
import glob
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#--------merge all excel files in a single csv file--------
folder_path= 'C:\\NARGES\\Trading\\data_row\\sol_eur\\SOLEUR_1m_2025_2024'
files = glob.glob(os.path.join(folder_path, '*.csv'))
print(f'Found {len(files)} CSV files in the folder.')
output_file = 'merged_data.csv'
##first_file = True  # Flag to write header only once
if os.path.exists(output_file):
    os.remove(output_file)
for file in files:
    print(f'Processing file: {file}')
    df = pd.read_csv(file, header=None)
    df.to_csv(output_file, mode='a', header=None, index=False)  # 'a'=append mode
    ##first_file = False  #after first write, don't write header again
print('merging is done!')
#--------add headers, convert date and time--------
cols= [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_base_vol", "taker_quote_vol", "ignore"]

df = pd.read_csv(
            output_file, header=None,
            names=cols, dtype={"open_time": "int64"}
                )

df = df.sort_values('open_time').reset_index(drop=True)
df['close'] = df['close'].astype(float)

def normalize_open_time(val):
    # If value is in microseconds (1e15+), convert to milliseconds
    if val > 1e14:
        return val // 1000
    # If value is in milliseconds (1e12+), keep as is
    elif val > 1e11:
        return val
    # If value is in seconds (1e9+), convert to milliseconds
    elif val > 1e8:
        return val * 1000
    else:
        return np.nan  # mark as invalid

df["open_time_norm"] = df["open_time"].apply(normalize_open_time)
df = df.dropna(subset=["open_time_norm"])  # remove invalid rows
df["datetime"] = pd.to_datetime(df["open_time_norm"], unit="ms", utc=True)
df['date'] = df['datetime'].dt.date
df['time'] = df['datetime'].dt.time
df = df.dropna(how='all')
print(df.head())
print(df.tail())
print('dataframe shape is: ', df.shape)
#------------------------------------------------------------------------------
#--------find 15% increas at price--------
from numba import njit
df1 = df[['close', 'date', 'time']]
prices = df1['close'].to_numpy(dtype=np.float64)
dates = df1['date'].to_numpy()
times = df1['time'].to_numpy()

@njit
def find_targets(prices, pct):
    n = len(prices)
    t_price = np.full(n, np.nan)
    t_index = np.full(n, -1)
    for i in range(n):
        target = prices[i] * (1.0 + pct)
        for j in range(i+1, n):
            if prices[j] >= target:
                t_price[i] = prices[j]
                t_index[i] = j
                break
    return t_price, t_index

# Run fast JIT loop
target_price, target_index = find_targets(prices, 0.15)

# Fill target date/time arrays
target_date = np.array([dates[j] if j != -1 else None for j in target_index], dtype=object)
target_time = np.array([times[j] if j != -1 else None for j in target_index], dtype=object)

# Create final DataFrame
result = pd.DataFrame({
    'price_buy': prices,
    'date_buy': dates,
    'time_buy': times,
    'target_price': target_price,
    'target_date': target_date,
    'target_time': target_time
})

print(result.head(40))
print(result.tail(40))
print(result.shape)
re_result = result.copy()
re_result = re_result.dropna(subset=['target_price'])
print(re_result.shape)
re_result['buy_datetime'] = pd.to_datetime(
        re_result['date_buy'].astype(str) + ' ' + re_result['time_buy'].astype(str)
                     )
re_result['target_datetime'] = pd.to_datetime(
        re_result['target_date'].astype(str) + ' ' + re_result['target_time'].astype(str)
        )
re_result['duration'] = re_result['target_datetime'] - re_result['buy_datetime']      
re_result['duration_days'] = re_result['duration'].dt.total_seconds() / 86400 
print(re_result.shape)
print('re_result is ready for next step!')
#-------------------------------------------------------------------------------------
#------------find max Loss in every buy and sell for 15% Profit
df1['datetime'] = pd.to_datetime(
         df1['date'].astype(str) + ' ' + df1['time'].astype(str))
print(df1.head())
import bisect

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        self.tree = np.full(2*self.size, np.inf) 
        self.tree[self.size:self.size+self.n] = data
        for i in range(self.size-1, 0, -1):
            self.tree[i] = min(self.tree[2*i], self.tree[2*i+1])

    def range_min(self, l, r):
        """Return min in [l, r)"""
        l += self.size
        r += self.size
        res = np.inf
        while l < r:
            if l % 2 == 1:
                res = min(res, self.tree[l])
                l += 1
            if r % 2 == 1:
                r -= 1
                res = min(res, self.tree[r])
            l //= 2
            r //= 2
        return res

df1 = df1.sort_values("datetime").reset_index(drop=True)
times = df1["datetime"].to_numpy()
prices = df1["close"].to_numpy()

# build segment tree on prices
seg = SegmentTree(prices)
from bisect import bisect_left, bisect_right

def get_min_price(buy_dt, target_dt):
    # find indices
    i = bisect_left(times, buy_dt)
    j = bisect_right(times, target_dt)
    if i >= j:  # no data in range
        return np.nan
    return seg.range_min(i, j)

max_losses = []
for _, row in re_result.iterrows():
    min_price = get_min_price(row["buy_datetime"], row["target_datetime"])
    if np.isnan(min_price):
        max_losses.append(np.nan)
    else:
        loss = (row["price_buy"]- min_price) / row["price_buy"] * 100
        max_losses.append(loss)

re_result["max_loss"] = max_losses
print('Its done')
print(re_result['max_loss'].head(20))
#--------------------------------------------------------------
#----------------plot max loss histogram
sns.histplot(re_result['max_loss'], edgecolor='black')
plt.title('Max Loss during each trade for 15% Profit')
plt.show()
#-------------------------------------------------------------------
#--------------find probability of max loss
Loss_probability = []
for i in range(3,50, 3):
    loss_i = re_result[re_result['max_loss'] <= i].shape[0]
    print(i, loss_i)
    percentage = loss_i / len(re_result) * 100
    l_p= [f' loss probability of {i} % is {percentage}']
    Loss_probability.append(l_p)
print(pd.DataFrame(Loss_probability))
#-------------------------------------------------------------------
