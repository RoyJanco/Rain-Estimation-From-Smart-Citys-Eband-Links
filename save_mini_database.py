import pickle
import pandas as pd
from matplotlib import pyplot as plt
from Smbit import Rehovot, RSL, load_rain_gauge

db_path = 'Data/db_2020.pickle'
db_path_save = 'Data/db_mini.pickle'
rg_path = 'Data/rain_gauge_20.xlsx'

hops_to_keep = [1, 2, 3, 4, 29]
date_start = '2020-11-01 00:00:00'
date_end = '2020-11-21 13:00:00'

with open(db_path, 'rb') as handle:
    db = pickle.load(handle)
    rg = load_rain_gauge(rg_path)

db_mini = {}
for hop in hops_to_keep:
    db_mini[hop] = db[hop]
    db_mini[hop].rsl = db_mini[hop].rsl[date_start:date_end]

# rg_mini = rg[date_start:date_end]

# Save database
with open(db_path_save, 'wb') as handle:
    print('Saving new database.')
    pickle.dump(db_mini, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load saved database
with open(db_path_save, 'rb') as handle:
    print('Loading saved database.')
    db_new = pickle.load(handle)

db_new[1].plot_rsl()
plt.show()

print('Done')

