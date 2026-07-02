########## Auxiliary - assisted by Claude ##########

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
import contextily as ctx
from pyproj import Transformer
from bidi.algorithm import get_display
from PIL import Image

# ── Save / Load ────────────────────────────────────────────────────────────────

def save_matrix(matrix, filename):
    with open(filename, 'wb') as f:
        pickle.dump(matrix, f)

def load_matrix(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

_BASE = os.path.dirname(os.path.abspath(__file__))
_df_religion = pd.read_excel(os.path.join(_BASE, '../data/cities_israel.xls'),
                              usecols=['symbol', 'דת יישוב']).rename(columns={'symbol': 'YeshuvKod'})
_RELIGION_MAP = {1: 'Jewish', 2: 'Arab', 3: 'Druze/Other', 4: 'Mixed'}

_CITY_FEATURES = ['population', 'מחוז', 'דת יישוב', 'מעמד מונציפאלי', 'גובה ']
_FEATURE_LABELS = ['Population', 'District', 'Religion', 'Municipal status', 'Elevation']
_df_features = pd.read_excel(os.path.join(_BASE, '../data/cities_israel.xls'),
                              usecols=['symbol'] + _CITY_FEATURES).rename(columns={'symbol': 'YeshuvKod'})

def load_and_merge():
    BASE = os.path.dirname(os.path.abspath(__file__))
    CRIMES_FILE = os.path.join(BASE, '../data/2012026154636_2025_hangashat_meyda_plili.xlsx')
    CITIES_FILE = os.path.join(BASE, '../data/cities_israel.xls')

    columns_to_read = ['YeshuvKod', 'StatisticGroupKod', 'StatisticTypeKod']
    df = pd.read_csv(CRIMES_FILE, usecols=columns_to_read, encoding='utf-8-sig')

    df_cities = pd.read_excel(CITIES_FILE, usecols=['symbol', 'coordinates', 'population'])
    df_cities = df_cities.rename(columns={'symbol': 'YeshuvKod'})

    # Pivot: rows = cities, columns = crime subtype, values = incident count
    df_counts = df.groupby(['YeshuvKod', 'StatisticTypeKod']).size().unstack(fill_value=0)

    df_merged = pd.merge(df_counts.reset_index(), df_cities, on='YeshuvKod', how='left')
    return df_merged

# ── Plotting ───────────────────────────────────────────────────────────────────

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../plots/unsupervised')
_itm_to_webmercator = Transformer.from_crs("EPSG:2039", "EPSG:3857", always_xy=True)

# labeled cities: (ITM_x/10, ITM_y/10, Hebrew name)
_LABELED_CITIES = [
    (19423, 38515, 'אילת'),
    (17966, 57357, 'באר שבע'),
    (22086, 63221, 'ירושלים'),
    (16691, 63377, 'אשדוד'),
    (18027, 66485, 'תל אביב-יפו'),
    (18726, 69145, 'נתניה'),
    (20112, 74546, 'חיפה'),
    (24942, 74369, 'טבריה'),
    (25383, 79074, 'קריית שמונה'),
]

def plot_cities(x, y, values, title, filename):
    mx, my = _itm_to_webmercator.transform(x * 10, y * 10)

    abs_max = np.percentile(np.abs(values), 95)
    norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)

    # Step 1: plot map content only (no title, no colorbar)
    fig, ax = plt.subplots(figsize=(8, 20))
    ax.scatter(mx, my, c=np.clip(values, -abs_max, abs_max), cmap='coolwarm', norm=norm, s=50, edgecolors='black',
               linewidths=0.3, alpha=0.9, zorder=3)
    for cx, cy, name in _LABELED_CITIES:
        wmx, wmy = _itm_to_webmercator.transform(cx * 10, cy * 10)
        # rotation=90 pre-compensates for the 270° image rotation so labels are upright
        ax.annotate(get_display(name), xy=(wmx, wmy), fontsize=12, rotation=90,
                    ha='center', va='bottom', zorder=4, color='#111111',
                    xytext=(0, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6, ec='none'))
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldGrayCanvas, zoom=8)
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    tmp_map = os.path.join(PLOTS_DIR, '_tmp_' + filename)
    plt.savefig(tmp_map, dpi=150, bbox_inches='tight')
    plt.close()

    # Step 2: rotate map 270° (west up)
    rotated = Image.open(tmp_map).rotate(270, expand=True)
    os.remove(tmp_map)

    # Step 3: compose final figure — rotated map + normal title + normal colorbar
    fig, (ax_map, ax_cb) = plt.subplots(1, 2, figsize=(20, 8),
                                         gridspec_kw={'width_ratios': [20, 1]})
    ax_map.imshow(rotated)
    ax_map.set_axis_off()
    ax_map.set_title(title, fontsize=16, pad=10)
    abs_max = np.percentile(np.abs(values), 95)
    norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    plt.colorbar(sm, cax=ax_cb)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()

def plot_eigengap(eigenvalues, n=20):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, n + 1), eigenvalues[1:n + 1], 'o-')
    ax.set_xlabel('Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Eigenvalue spectrum (eigengap)')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'eigengap.png'), dpi=150)
    plt.close()

def print_correlation_table(labels, rows):
    print(f"{'Feature':<20}", end='')
    for label in labels:
        print(f"  {label:>6}", end='')
    print()
    print('-' * (20 + 8 * len(labels)))
    for feat_label, values, stars in rows:
        print(f"{feat_label:<20}", end='')
        for r, s in zip(values, stars):
            print(f"  {r:+.2f}{s}", end='')
        print()
    print()

def plot_clusters(x, y, labels, title, filename):
    mx, my = _itm_to_webmercator.transform(x * 10, y * 10)
    k = len(np.unique(labels))
    cmap = plt.cm.get_cmap('tab10', k)

    # Step 1: plot map content only
    fig, ax = plt.subplots(figsize=(8, 20))
    ax.scatter(mx, my, c=labels, cmap=cmap, vmin=0, vmax=k - 1,
               s=50, edgecolors='black', linewidths=0.3, alpha=0.9, zorder=3)
    for cx, cy, name in _LABELED_CITIES:
        wmx, wmy = _itm_to_webmercator.transform(cx * 10, cy * 10)
        ax.annotate(get_display(name), xy=(wmx, wmy), fontsize=12, rotation=90,
                    ha='center', va='bottom', zorder=4, color='#111111',
                    xytext=(0, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6, ec='none'))
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldGrayCanvas, zoom=8)
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    tmp_map = os.path.join(PLOTS_DIR, '_tmp_' + filename)
    plt.savefig(tmp_map, dpi=150, bbox_inches='tight')
    plt.close()

    # Step 2: rotate 270°
    rotated = Image.open(tmp_map).rotate(270, expand=True)
    os.remove(tmp_map)

    # Step 3: compose with title and discrete colorbar
    fig, (ax_map, ax_cb) = plt.subplots(1, 2, figsize=(20, 8),
                                         gridspec_kw={'width_ratios': [20, 1]})
    ax_map.imshow(rotated)
    ax_map.set_axis_off()
    ax_map.set_title(title, fontsize=18)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=k - 1))
    cbar = plt.colorbar(sm, cax=ax_cb, ticks=range(k))
    cbar.set_ticklabels([f'Cluster {i+1}' for i in range(k)])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()


########## Spectral Analysis ##########

## Preprocess
def preprocess(df_merged):
    # drop rows with unknown settlement
    df_merged = df_merged.dropna(subset=['YeshuvKod'])

    # drop prblematic data categories as mentioned inthe writeup: -1, 1200, 1300
    BAD_GROUPS = {-1, 1200, 1300}
    crime_cols_all = [c for c in df_merged.columns if isinstance(c, int) and c not in BAD_GROUPS]
    df_merged = df_merged[['YeshuvKod'] + crime_cols_all + ['coordinates', 'population']]

    # keep cities with known population and coords
    df_merged['population'] = pd.to_numeric(df_merged['population'], errors='coerce')
    df_merged['coordinates'] = pd.to_numeric(df_merged['coordinates'], errors='coerce')
    df_merged = df_merged.dropna(subset=['population', 'coordinates'])
    df_merged = df_merged[df_merged['population'] >= 2000]
    df_merged = df_merged.set_index('YeshuvKod')

    # Parse ITM coordinate integer into x (easting) and y (northing)
    coord_str = df_merged['coordinates'].astype(int).astype(str).str.zfill(10)
    df_merged['x'] = coord_str.str[:5].astype(float)
    df_merged['y'] = coord_str.str[5:10].astype(float)
    df_merged = df_merged.drop(columns=['coordinates'])

    crime_cols = [c for c in df_merged.columns if isinstance(c, int)]
    print(f'Cities after preprocessing: {len(df_merged)}')
    print(f'Crime subtypes: {len(crime_cols)}')
    return df_merged, crime_cols

## Step 3 - compute Laplacian

# Create embedding matrix:
def compute_embedding_matrix(C, pop):
    # noarmlize by population, resphape to (|crime_cols|, 1)
    V = C / pop[:, np.newaxis]
    # L2 normalization per row, so we compare crime mix profile
    row_norms = np.linalg.norm(V, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    V = V / row_norms
    return V

# calculating sigma based on k-nearest neighbor like recommended in "A Tutorial on Spectral Clustering' paper we learned
def compute_sigmas(V, k=7):
    n = V.shape[0]
    sigmas = np.zeros(n)
    for i in range(n):
        dists = np.array([np.sum((V[i] - V[j]) ** 2) ** 0.5 for j in range(n) if j != i])
        dists.sort()
        sigmas[i] = dists[k - 1]
    return sigmas

# Compute weight matrix W:
def compute_LW_matrices(V, sigmas, n):
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            diff = V[i]-V[j]
            # Kernel
            dist_squared =np.sum(diff**2)
            W[i,j] = np.exp(-dist_squared / (sigmas[i] * sigmas[j]))
            W[j,i] = W[i,j]
    np.fill_diagonal(W, 0)
    # Copmute L and D
    D= np.zeros((n,n))
    for i in range(n):
        cur_sum =0
        for j in range(n):
            cur_sum+= W[i,j]
        D[i,i] =  cur_sum
    L = D-W
    return L,W

## Step 4 - EV calculation
def calc_print_EV(L, x, y, city_names):
    # get eigenvectors and eigenvalues
    eigenvals, eigenvecs = np.linalg.eigh(L)

    for t in range(1, 11): # skipping the trivial 0-eigenvalue vector:
        ev = eigenvecs[:, t]
        title = f'Eigenvector {t} (λ={eigenvals[t]:.4f})'
        plot_cities(x, y, ev, title=title, filename=f'eigenvector_{t}.png')
    return eigenvals, eigenvecs

## Step 5 - spectral clustering
def spectral_clustering(eigenvecs, k):
    v = eigenvecs[:, 1:k + 1]
    kmeans = KMeans(n_clusters=k, random_state=23)
    labels = kmeans.fit_predict(v)
    return labels

## Step 6 - analyzing ethnicity and correlate eigenvectors with city features
def analyze_ethnicity(clusters, yeshuv_kods):
    # pair city with its kod, join the religion df
    df = pd.DataFrame({'cluster': clusters, 'YeshuvKod': yeshuv_kods})
    df = df.merge(_df_religion, on='YeshuvKod', how='left')
    # convert codes to labels
    df['religion'] = df['דת יישוב'].map(_RELIGION_MAP).fillna('Unknown')
    # count cluster-religion pairs
    table = df.groupby(['cluster', 'religion']).size().unstack(fill_value=0)
    table.index = [f'Cluster {i+1}' for i in table.index]
    print(table.to_string())
    print()

def correlate_features(eigenvecs, yeshuv_kods):
    # load df
    df = pd.DataFrame({'YeshuvKod': yeshuv_kods}).merge(_df_features, on='YeshuvKod', how='left')
    ev_labels = [f'EV{t}' for t in range(1, 6)]
    rows = []
    for col, label in zip(_CITY_FEATURES, _FEATURE_LABELS):
        feat = df[col].values
        # mask missing values from correlation computation
        mask = ~np.isnan(feat)
        rs, stars = [], []
        # calculate spearman
        for t in range(1, 6):
            r, p = spearmanr(eigenvecs[mask, t], feat[mask])
            rs.append(r)
            # check for significance ( uncorrected for BH like in next part)
            stars.append('*' if p < 0.05 else ' ')
        rows.append((label, rs, stars))
    print_correlation_table(ev_labels, rows)

def main():
    df_merged, crime_cols = preprocess(load_and_merge())
    # step 3
    n = len(df_merged)
    C = df_merged[crime_cols].values.astype(float)
    pop = df_merged['population'].values
    V = compute_embedding_matrix(C, pop)
    sigmas = compute_sigmas(V)
    L, W = compute_LW_matrices(V, sigmas, n)
    save_matrix(W, 'W.pkl')
    save_matrix(L, 'L.pkl')
    # step 4
    x, y = df_merged['x'].values, df_merged['y'].values
    BASE = os.path.dirname(os.path.abspath(__file__))
    df_cities = pd.read_excel(os.path.join(BASE, '../data/cities_israel.xls'),
                              usecols=['symbol', 'שם יישוב'])
    kod_to_name = dict(zip(df_cities['symbol'], df_cities['שם יישוב']))
    city_names = [kod_to_name.get(k, str(int(k))) for k in df_merged.index]
    eigenvals, eigenvecs = calc_print_EV(L, x, y, city_names)
    # step 5
    plot_eigengap(eigenvals)
    clusters = spectral_clustering(eigenvecs, 5)
    plot_clusters(x, y, clusters, title='Spectral Clustering (k=5)', filename='clusters.png')
    # step 6
    analyze_ethnicity(clusters, list(df_merged.index))
    correlate_features(eigenvecs, list(df_merged.index))

if __name__ == '__main__':
    main()
