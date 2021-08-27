import pandas as pd
import matplotlib.pyplot as plt

time_data = pd.read_excel("./all_training_times.xlsx")

x = [10, 50, 250, 500, 1300]

y_tradi = []
y_nssdl = []
y_nssdl_strong = []
y_mixtext = []

for data_class in x:
    for index, row in time_data.iterrows():
        if row['data_class'] == data_class:
            if row['model'] == 'tradi':
                y_tradi.append(row['time'])
            if row['model'] == 'nssdl':
                y_nssdl.append(row['time'])
            if row['model'] == 'nssdl_strong':
                y_nssdl_strong.append(row['time'])
            if row['model'] == 'mixtext':
                y_mixtext.append(row['time'])

plt.plot(x, y_tradi, label='Traditional')
plt.plot(x, y_nssdl, label='NSSDL')
plt.plot(x, y_nssdl_strong, label='NSSDL Strong')
plt.plot(x, y_mixtext, label='MixText')

plt.xlabel('Dataclass')
plt.ylabel('Average Train time in seconds')
plt.legend()
plt.show()
plt.savefig(f'./training_time_graph')