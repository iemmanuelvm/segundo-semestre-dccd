import numpy as np
import pandas as pd

np.random.seed(0)
num_channels = 20
num_points = 1000

time = np.linspace(0, 10, num_points)

data = {f'Channel_{i+1}': np.random.normal(0, 50, num_points) for i in range(num_channels)}
data['Time'] = time

df = pd.DataFrame(data)
df.to_csv('signals.csv', index=False)

print('Signals generated and saved to signals.csv')
