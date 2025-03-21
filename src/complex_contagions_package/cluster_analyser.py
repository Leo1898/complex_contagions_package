import numpy as np
import xarray as xr
from sklearn.mixture import GaussianMixture
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Beispielhafte exponentielle Sättigungsfunktion
def exp_saturation(alpha, a, b, c):
    return a * alpha**b * np.exp(-c * alpha)

# 1. Daten laden (Hier als Platzhalter)
# dataset = xr.open_dataset("your_file.nc")
# alpha_values = dataset["alpha"].values  # (50,)
# hyst_area_values = dataset["hyst_area"].values  # (50, 100)


# Lade das xarray-Dataset (hier als Platzhalter mit Zufallswerten generiert)
def load_dataset():
    alpha = np.linspace(2, 100, 50)
    simulation = np.arange(1, 101)
    t0 = np.linspace(0, 1, 101)
    steps = np.arange(1, 51)
    data = np.random.rand(len(alpha), len(simulation), len(t0), len(steps))
    return xr.Dataset({"inflist_asc": (['alpha', 'simulation', 't0', 'steps'], data)})

data = load_dataset()

# Extrahiere relevante Daten für die Clusteranalyse
alpha_values = data.coords['alpha'].values
hysteresis_values = data['inflist_asc'].mean(dim=['simulation', 't0', 'steps']).values  # Mitteln über Dimensionen

# GMM für Clusterbildung der Sättigungsfunktionen
n_clusters = 3  # Hypothetische Clusteranzahl
X = np.column_stack((alpha_values, hysteresis_values))
gmm = GaussianMixture(n_components=n_clusters, covariance_type='full')
gmm.fit(X)
labels = gmm.predict(X)

# Curve Fitting für jedes Cluster
cluster_params = []
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
for i in range(n_clusters):
    cluster_indices = labels == i
    popt, _ = curve_fit(exp_saturation, alpha_values[cluster_indices], hysteresis_values[cluster_indices], maxfev=10000)
    cluster_params.append(popt)
    plt.scatter(alpha_values[cluster_indices], hysteresis_values[cluster_indices], label=f'Cluster {i+1}', alpha=0.6, color=colors[i])
    plt.plot(alpha_values, exp_saturation(alpha_values, *popt), linestyle='--', color=colors[i])

plt.xlabel('Alpha')
plt.ylabel('Hysteresis Area')
plt.legend()
plt.title('Clusteranalyse mit exponentiellen Sättigungsfunktionen')
plt.show()

# Die ermittelten Parameter der Cluster
for i, params in enumerate(cluster_params):
    print(f'Cluster {i+1} Parameter: a={params[0]:.3f}, b={params[1]:.3f}, c={params[2]:.3f}')

