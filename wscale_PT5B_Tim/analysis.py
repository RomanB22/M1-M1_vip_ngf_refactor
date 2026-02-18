import pandas
import ast
from scipy.interpolate import interp1d
import pickle


def parse_sec_loc(sec_loc):
    return ast.literal_eval(sec_loc)


EPSPNORM = 0.5
df = pandas.read_csv('grid_search.csv')[['config/sec', 'config/weight', 'epsp']]
sec_locs = [[i, 0.5] for i in df['config/sec']]
df[['sec', 'loc']] = pandas.DataFrame(sec_locs, index=df.index)
secs = df['sec'].unique()

wnorms = {}

for sec in secs:
    # for each section calculate the weight where the epsp at soma == 0.5
    entries = df[df['config/sec'] == sec][['config/weight', 'epsp']].dropna()
    entries = entries.sort_values(by='epsp')
    entries = entries.drop_duplicates(subset='epsp', keep='last')

    if len(entries) < 2:
        continue

    epsps = entries['epsp'].to_numpy()
    weights = entries['config/weight'].to_numpy()
    f = interp1d(epsps, weights, fill_value='extrapolate', bounds_error=False)
    w = float(f(EPSPNORM))
    if w <= 0:
        positive_weights = entries[entries['config/weight'] > 0]['config/weight']
        if len(positive_weights) == 0:
            continue
        w = float(positive_weights.min())

    wnorm = w / EPSPNORM
    wnorms[sec] = [wnorm]

filename = 'PT5B_full_weightNorm_TIM.pkl' # 'weight_norms.pkl'
print(wnorms)
with open(filename, 'wb') as fptr:
    pickle.dump(wnorms, fptr)
