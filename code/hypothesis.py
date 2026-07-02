########## Auxiliary - assisted by Claude ##########

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
from scipy import stats

_BASE = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(_BASE, '../plots/hypothesis')

BAD_GROUPS = {-1, 1200, 1300}

_CITIES_FILE = os.path.join(_BASE, '../data/cities_israel.xls')

# ── Data loading ───────────────────────────────────────────────────────────────

def _load_one_file(filepath):
    cols = ['Year', 'YeshuvKod', 'StatisticGroupKod', 'StatisticTypeKod', 'StatisticType']
    # try csv first; fall back to xls/xlsx
    try:
        df = pd.read_csv(filepath, usecols=cols, encoding='utf-8-sig')
    except Exception:
        df = pd.read_excel(filepath, usecols=cols)
    return df

def load_yearly_rates(filepaths):
    # load files, filter bad groups and unknown cities, join population >= 2000
    frames = [_load_one_file(p) for p in filepaths]
    df = pd.concat(frames, ignore_index=True)
    df = df[df['Year'].isin([2022, 2025])]
    df = df[~df['StatisticGroupKod'].isin(BAD_GROUPS)]
    df = df.dropna(subset=['YeshuvKod'])

    df_cities = pd.read_excel(_CITIES_FILE, usecols=['symbol', 'population'])
    df_cities = df_cities.rename(columns={'symbol': 'YeshuvKod'})
    df_cities['population'] = pd.to_numeric(df_cities['population'], errors='coerce')
    df_cities = df_cities[df_cities['population'] >= 2000]

    df = df.merge(df_cities[['YeshuvKod', 'population']], on='YeshuvKod', how='inner')
    return df

# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_pvalues(df, bonf_threshold):
    # df is sorted output of run_t_test_with_bh (ascending by p_value)
    # style follows Ex2/q2.py: rank on x, p-value on log-scale y, horizontal threshold lines
    m = len(df)
    sorted_p = df['p_value'].values  # already sorted ascending
    bh_thresh = df.loc[df['reject_bh'], 'p_value'].max() if df['reject_bh'].any() else 0
    bonf_thresh = bonf_threshold

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, m + 1), sorted_p, marker='.', markersize=4, linewidth=0.8,
            color='steelblue', label='p-values')
    ax.axhline(bonf_thresh, color='orange', linestyle='--',
               label=f'Bonferroni ({bonf_thresh:.5f})')
    ax.axhline(bh_thresh, color='black', linestyle='--',
               label=f'BH threshold ({bh_thresh:.5f})')
    # list significant crime types in bottom-right as a text block
    sig = df[df['reject_bh']].reset_index(drop=True)
    label_text = 'Significant crime types:\n' + '\n'.join(
        get_display(f"{i+1}. {row['StatisticType']}")
        for i, row in sig.iterrows()
    )
    ax.text(0.98, 0.02, label_text, transform=ax.transAxes,
            fontsize=7, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='lightgrey'))
    ax.set_yscale('log')
    ax.set_xlabel('Rank')
    ax.set_ylabel('p-value')
    ax.set_title('2022 vs 2025: crime type p-values')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'pvalues.png'), dpi=150)
    plt.close()


########## Multiple Hypothesis Testing ##########

def compute_city_rates(df):
    # aggregate (type, city, year)
    counts = (
        df.groupby(['StatisticTypeKod', 'StatisticType', 'YeshuvKod', 'population', 'Year'])
        .size()
        .rename('crime_count')
        .reset_index()
    )
    # compute per-capita crime rate
    counts['crime_rate'] = counts['crime_count'] / counts['population']
    return counts

def run_t_test_with_bh(counts, alpha=0.05):
    results = []
    for type_kod, group in counts.groupby('StatisticTypeKod'):
        # each entry is one city's per-capita rate for the specific crime type
        rates_2022 = group[group['Year'] == 2022]['crime_rate'].values
        rates_2025 = group[group['Year'] == 2025]['crime_rate'].values

        # skip rare types with too few cities to estimate variance
        if len(rates_2022) < 2 or len(rates_2025) < 2:
            continue

        #calculating welch t_test statistic and accordingly the pval, without assuming equal varaince
        T_i, p_i = stats.ttest_ind(rates_2022, rates_2025, equal_var=False)
        results.append({
            'StatisticTypeKod': type_kod, 'StatisticType': group['StatisticType'].iloc[0], 'mean_2022': rates_2022.mean(),
            'mean_2025': rates_2025.mean(), 'effect': rates_2025.mean() - rates_2022.mean(), 'T_i': T_i, 'p_value': p_i,
        })

    df = pd.DataFrame(results)
    # calc bonferroni
    m = len(df)
    bonf_threshold = alpha / m

    ##  calc BH:
    # sort pvals:
    df = df.sort_values('p_value').reset_index(drop=True)
    # calc bf threshold
    df['bh_threshold'] = (df.index + 1) * alpha / m
    # find k corresponding to threshold
    k = df[df['p_value'] <= df['bh_threshold']].index.max()
    # reject all hypotheses with rank <= k_max
    df['reject_bh'] = False
    if not pd.isna(k):
        df.loc[:k, 'reject_bh'] = True

    df['reject_bonferroni'] = df['p_value'] < bonf_threshold

    return df, bonf_threshold


def main():
    import glob
    files = sorted(glob.glob(os.path.join(_BASE, '../data/2012026154636_2022*.csv'))) + \
            sorted(glob.glob(os.path.join(_BASE, '../data/2012026154636_2025*.xlsx')))
    counts = compute_city_rates(load_yearly_rates(files))
    df, bonf_threshold = run_t_test_with_bh(counts)
    plot_pvalues(df, bonf_threshold)

if __name__ == '__main__':
    main()
