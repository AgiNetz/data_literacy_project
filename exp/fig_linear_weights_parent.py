import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from tueplots import bundles

plt.rcParams.update(bundles.neurips2021(usetex=False))
import sys
import json
import os
sys.path.insert(0, os.path.realpath('..'))
os.chdir('..')

from src import data
from src.model import get_splits

with open('config.json', 'r') as cfg:
    config = json.load(cfg)

random_seed = 42

dataset = data.load_data(config, False)
clean_data = data.filter_bad_data(dataset)
samples_pcnt = data.create_samples(clean_data)[0]

train, test = get_splits(samples_pcnt, test_size=config["test_size"], random_seed=random_seed)

train['Total expenditure per capita (1000s USD)'] /= 1000

train_fts = train.iloc[:,2:-2].to_numpy()

train_labels = train["Happiness score"].to_numpy()

model = LinearRegression(fit_intercept=False).fit(train_fts, train_labels)

named_weights = clean_data.groupby(['Function code', 'Function'], as_index=False).sum()[['Function code', 'Function']].set_index('Function code')
named_weights["Weight"] = model.coef_

parent_data = data.load_data(config, True)

parent_code_mapping = parent_data.groupby(['Function code', 'Function'], as_index=False).sum()[['Function code', 'Function']].set_index('Function code')

ax = plt.gca()
#fig = plt.gcf()
#fig.set_figheight(10)
#fig.set_figwidth(20)
ax.bar(named_weights["Function"], named_weights["Weight"])
ax.set_xticklabels(named_weights["Function"], rotation = 90)
plt.savefig('linear_weights.pdf', format='pdf')
