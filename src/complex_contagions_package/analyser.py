"""Simulation analysis."""
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr

# from complex_contagions_package.logging import log_error
# from complex_contagions_package.simulator import run_hysteresis_simulation

def analyze_clusters(G):
    """Analysiert Cluster infizierter Knoten im Netzwerk G."""
    infected_nodes = [n for n, d in G.nodes(data=True) if d.get("inf", 0) == 1]
    subgraph = G.subgraph(infected_nodes)
    clusters = list(nx.connected_components(subgraph))
    cluster_sizes = [len(c) for c in clusters]

    if not cluster_sizes:
        return {
            "max_cluster_size": 0,
            "mean_cluster_size": 0,
            "largest_cluster_fraction": 0,
            "n_clusters": 0
        }

    max_cluster = max(cluster_sizes)
    return {
        "max_cluster_size": max_cluster,
        "mean_cluster_size": np.mean(cluster_sizes),
        "largest_cluster_fraction": max_cluster / len(infected_nodes),
        "n_clusters": len(cluster_sizes)
    }

def consolidate_data(simulation, ds):
    """Returns data for chosen alpha and simulation no. or average if selected."""
    if isinstance(simulation, str) and simulation == "Average":
        inflist_asc_last_step = ds.inflist_asc.isel(steps=-1).mean(
            dim="simulation"
            )
        inflist_desc_last_step = ds.inflist_desc.isel(steps=-1).mean(
            dim="simulation"
            )
        inflist_asc_all_steps = ds.inflist_asc.mean(
            dim="simulation"
            )
        inflist_desc_all_steps = ds.inflist_desc.mean(
            dim="simulation"
            )
    else:
        simulation_index = simulation - 1
        inflist_asc_last_step = ds.inflist_asc.isel(
            simulation=simulation_index, steps=-1
            )
        inflist_desc_last_step = ds.inflist_desc.isel(
            simulation=simulation_index, steps=-1
            )
        inflist_asc_all_steps = ds.inflist_asc.isel(
            simulation=simulation_index
            )
        inflist_desc_all_steps = ds.inflist_desc.isel(
            simulation=simulation_index
            )


    return (inflist_asc_last_step, inflist_desc_last_step,
            inflist_asc_all_steps, inflist_desc_all_steps)

def hysteresis_calc(simulation, ds,
                    alphas,
                    t0):
    """Calculates hysteresis areas and averages per alpha."""
    #df_t0 = []
    df_t0agg = []
    df_simagg = []

    # cluster_vars_exist = (
    #     "max_cluster_size_asc" in ds.variables and
    #     "mean_cluster_size_asc" in ds.variables and
    #     "largest_cluster_fraction_asc" in ds.variables and
    #     "n_clusters_asc" in ds.variables
    # )

    for alpha in alphas:
        areas_per_alpha = []

        for sim in range(len(simulation)):

            # # Infektionsdifferenz am letzten Step
            # I_up = ds["inflist_asc"].sel(alpha=alpha).isel(simulation=sim, steps=-1)
            # I_down = ds["inflist_desc"].sel(alpha=alpha).isel(simulation=sim, steps=-1)
            # H = I_up - I_down

            # if cluster_vars_exist:
            #     # MAX_Clusterdifferenz
            #     CMax_up = ds["max_cluster_size_asc"].sel(alpha=alpha).isel(simulation=sim)
            #     CMax_down = ds["max_cluster_size_desc"].sel(alpha=alpha).isel(simulation=sim)
            #     CMaxdiff = CMax_up - CMax_down

            #     # MEAN_Clusterdifferenz
            #     CMean_up = ds["mean_cluster_size_asc"].sel(alpha=alpha).isel(simulation=sim)
            #     CMean_down = ds["mean_cluster_size_desc"].sel(alpha=alpha).isel(simulation=sim)
            #     CMeandiff = CMean_up - CMean_down

            #     # CFracMax_Clusterdifferenz
            #     CFracMax_up = ds["largest_cluster_fraction_asc"].sel(alpha=alpha).isel(simulation=sim)
            #     CFracMax_down = ds["largest_cluster_fraction_desc"].sel(alpha=alpha).isel(simulation=sim)
            #     CFracMaxdiff = CFracMax_up - CFracMax_down

            #     # nC_Clusterdifferenz
            #     nC_up = ds["n_clusters_asc"].sel(alpha=alpha).isel(simulation=sim)
            #     nC_down = ds["n_clusters_desc"].sel(alpha=alpha).isel(simulation=sim)
            #     nCdiff = nC_up - nC_down
            # else:
            #     # Platzhalter np.nan, wenn Clusterdaten fehlen
            #     CMaxdiff = np.full_like(H, np.nan)
            #     CMeandiff = np.full_like(H, np.nan)
            #     CFracMaxdiff = np.full_like(H, np.nan)
            #     nCdiff = np.full_like(H, np.nan)

            # for i in range(len(t0)):
            #     df_t0.append({"alpha": alpha,
            #                     "simulation": sim+1,
            #                     "t0": t0[i].item(),
            #                     "H": H[i].item(),
            #                     "CMaxdiff": CMaxdiff[i].item(),
            #                     "CMeandiff": CMeandiff[i].item(),
            #                     "CFracMaxdiff": CFracMaxdiff[i].item(),
            #                     "nCdiff": nCdiff[i].item()
            #                 })

            asc_sim = ds.inflist_asc.sel(alpha=alpha).isel(simulation=sim, steps =-1)
            desc_sim = ds.inflist_desc.sel(alpha=alpha).isel(simulation=sim, steps =-1)
            area = np.trapz((asc_sim - desc_sim), x=t0)
            df_t0agg.append({"alpha": alpha,
                            "simulation": sim+1, #1-based
                            "hysteresis_area": area
                                    })
            areas_per_alpha.append(area)

        avg_area = np.mean(areas_per_alpha)
        df_simagg.append({"alpha": alpha,
                          "avg_hysteresis_area": avg_area
                        })

    #df_t0 =  pd.DataFrame(df_t0)
    df_t0agg = pd.DataFrame(df_t0agg)
    df_simagg = pd.DataFrame(df_simagg)

    # df = (
    # df_t0
    # .merge(df_t0agg, how="left", on=["alpha", "simulation"])
    # .merge(df_simagg, how="left", on=["alpha"])
    # )

    #return df_t0, df_t0agg, df_simagg
    return df_t0agg, df_simagg

def calculate_peak_and_t0(simulation, ds, alphas, t0):
    """Calculates the maximum peak and the corresponding average t0.

     For each simulation,as well as their mean values per alpha.

    Parameters:
    - simulation: Simulation IDs (list of integers or "Average").
    - ds (xarray.Dataset): The dataset containing simulation data.
    - alphas (array-like): Array of alpha values.
    - t0 (array-like): Array of t0 values.

    Returns:
    - peaks_df (pd.DataFrame): DataFrame with columns for alpha, simulation, max peak,
                               and corresponding t0.
    - mean_peaks_df (pd.DataFrame): DataFrame with mean max_peak and peak_t0 per alpha.
    """
    inflist_asc, inflist_desc, _, _ = consolidate_data(simulation, ds)

    peaks_data = []

    for alpha in alphas:
        asc_curve = inflist_asc.sel(alpha=alpha)
        desc_curve = inflist_desc.sel(alpha=alpha)

        # Für alle Simulationen die Maxima berechnen
        num_simulations = len(simulation)
        for sim_idx in range(num_simulations):
            asc_sim = asc_curve.isel(simulation=sim_idx)
            desc_sim = desc_curve.isel(simulation=sim_idx)

            distance = np.abs(asc_sim - desc_sim)

            # Maximalen Abstand und zugehörige t0-Werte finden
            max_distance = distance.max().item()
            peak_indices = np.where(distance.values == max_distance)[0]
            peak_t0_values = t0[peak_indices]

            # Ergebnisse speichern
            peaks_data.append({
                "alpha": alpha,
                "simulation": sim_idx + 1,  # Simulationen 1-basiert nummerieren
                "max_peak": max_distance,
                "peak_t0": np.mean(peak_t0_values)  # Mittelwert bei mehrfachen Peaks
            })

    # DataFrame erstellen
    peaks_df = pd.DataFrame(peaks_data)

    # Durchschnittswerte pro Alpha berechnen
    mean_peaks_df = peaks_df.groupby("alpha").agg({
        "max_peak": "mean",
        "peak_t0": "mean"
    }).reset_index()

    # Beide DataFrames zurückgeben
    return peaks_df, mean_peaks_df

def find_all_critical_t0_from_list(inflist_ascending, hys_threshold):
    """Findet alle kritischen t0-Werte (Sprünge) basierend auf der Endinfektionsrate."""
    critical_t0_values = []
    previous_value = inflist_ascending[0][2][-1]  # letzter step von erstem t0

    for idx, t0, inflist in inflist_ascending:
        current_value = inflist[-1]  # letzter step dieser t0-Simulation
        if abs(current_value - previous_value) > hys_threshold:
            critical_t0_values.append((idx, t0, current_value))
        previous_value = current_value

    return critical_t0_values

def is_stable_within_window(infectionlist, index, hys_threshold, window_size=30):
    """Prüft, ob die Infektionswerte in einem Fenster stabil sind."""
    end_index = min(index + window_size, len(infectionlist))
    window_values = [infectionlist[i][2][-1] for i in range(index, end_index)]
    return all(abs(window_values[i] - window_values[i - 1]) < hys_threshold
               for i in range(1, len(window_values)))

def find_stable_critical_t0(inflist_ascending, hys_threshold):
    """Findet den stabilen kritischen t0-Wert aus der Liste inflist_ascending."""
    criticals = find_all_critical_t0_from_list(inflist_ascending, hys_threshold)

    if not criticals:
        return None

    for (idx, t0, _) in reversed(criticals):
        if is_stable_within_window(inflist_ascending, idx, hys_threshold):
            return t0  # stabiler kritischer t0 gefunden

    return criticals[-1][1]  # fallback: letzter (nicht stabiler) kritischer t0


# def find_all_critical_t0(ds, hys_threshold):
#     """Identifies critical t0 values directly from the dataset.

#     Parameters:
#     - ds (xarray.Dataset): The dataset containing simulation data.
#     - hys_threshold (float): Jump in end infections defined to be critical.

#     Returns:
#     - critical_t0_values_asc (dict): Critical t0 values for ascending order,
#                                      grouped by alpha.
#     - critical_t0_values_desc (dict): Critical t0 values for descending order,
#                                       grouped by alpha.
#     """
#     critical_t0_values_asc = {}
#     critical_t0_values_desc = {}

#     # Iteriere über alle alpha-Werte
#     for alpha in ds.alpha.values:
#         # Daten für das aktuelle alpha extrahieren
#         inflist_asc = ds.sel(alpha=alpha).inflist_asc
#         inflist_desc = ds.sel(alpha=alpha).inflist_desc

#         # Initialisierung für aufsteigende Reihenfolge
#         critical_asc = []
#         previous_value_asc = inflist_asc.isel(steps=-1).isel(t0=0).mean(
#             dim="simulation").item()

#         # Über alle t0 in aufsteigender Reihenfolge iterieren
#         for idx, t0 in enumerate(ds.t0.values):
#             current_value = inflist_asc.isel(steps=-1).sel(t0=t0).mean(
#                 dim="simulation").item()
#             if abs(current_value - previous_value_asc) > hys_threshold:
#                 critical_asc.append((idx, t0, current_value))
#             previous_value_asc = current_value

#         # Speichere kritische Werte für ascending
#         critical_t0_values_asc[alpha] = critical_asc

#         # Initialisierung für absteigende Reihenfolge
#         critical_desc = []
#         previous_value_desc = inflist_desc.isel(steps=-1).isel(t0=0).mean(
#             dim="simulation").item()

#         # Über alle t0 in absteigender Reihenfolge iterieren
#         for idx, t0 in enumerate(ds.t0.values):
#             current_value = inflist_desc.isel(steps=-1).sel(t0=t0).mean(
#                 dim="simulation").item()
#             if abs(current_value - previous_value_desc) > hys_threshold:
#                 critical_desc.append((idx, t0, current_value))
#             previous_value_desc = current_value

#         # Speichere kritische Werte für descending
#         critical_t0_values_desc[alpha] = critical_desc

#     return critical_t0_values_asc, critical_t0_values_desc


# def is_stable_within_window(infectionlist, index, hys_threshold, window_size = 3):
#     """Check if the infection rates within a sliding window are stable.

#     Ensures the window is within the bounds of the infectionlist, collects values within
#     the window and checks if all values in the window are within the hys_threshold.

#     Parameters:
#     - infectionlist: infected nodes results for asc/desc t0 values
#     - index: current index
#     - hys_threshold: jump of end infections defined to be critical (set in config)
#     """
#     end_index = min(index + window_size, len(infectionlist))

#     window_values = [infectionlist[i][2][-1] for i in range(index, end_index)]

#     return all(abs(window_values[i] - window_values[i-1]) < hys_threshold
#                for i in range(1, len(window_values)))

# def find_stable_critical_t0(inflist_ascending, inflist_descending, hys_threshold):
#     """Find stable critical points for ascending and descending t0 values.

#     Gets critical t0 values for ascending and descending order. Returns None if
#     no critical values were found or returns both stable points if stable values
#     within window were found (or last critical jump if none is stable).

#     Parameters:
#     - inflist_ascending: infected nodes results for ascending t0 values
#     - inflist_descending: infected nodes results for descending t0 values
#     - hys_threshold: jump of end infections defined to be critical (set in config)
#     """
#     critical_t0_values_asc, critical_t0_values_desc = find_all_critical_t0(
#         inflist_ascending,
#         inflist_descending,
#         hys_threshold
#         )

#     if not critical_t0_values_asc and not critical_t0_values_desc:
#         return None

#     stable_asc = None
#     for (idx, t0, _) in reversed(critical_t0_values_asc):
#         if is_stable_within_window(inflist_ascending, idx, hys_threshold):
#             stable_asc = t0
#             break

#     stable_desc = None
#     for (idx, t0, _) in reversed(critical_t0_values_desc):
#         if is_stable_within_window(inflist_descending, idx, hys_threshold):
#             stable_desc = t0
#             break

#     return stable_asc or (critical_t0_values_asc[-1][1] if
#                           critical_t0_values_asc else None), \
#            stable_desc or (critical_t0_values_desc[-1][1] if
#                            critical_t0_values_desc else None)

# def calculate_hysteresis_gaps(
#         hys_threshold, alpha, g_type,
#         t0_values_ascending, t0_values_descending,
#         inf_chance, rec_chance, steps):
#     """Identifies hysteresis if present.

#   Runs a simulation, checks if valid values are returned, calls stable critical values
#     and calculates and returns the differnce between stable critical values as well as
#     infection results of the simulation.

#     Parameters:
#     - inflist_ascending: infected nodes results for ascending t0 values
#     - inflist_descending: infected nodes results for descending t0 values
#     - hys_threshold: jump of end infections defined to be critical (set in Config)
#     """
#     hysteresis_gaps = []

#     inflist_ascending, inflist_descending = run_hysteresis_simulation(
#         g_type, steps,
#         t0_values_ascending, t0_values_descending,
#         inf_chance, rec_chance,
#         )

#     if any(x is None or not x for x in (inflist_ascending, inflist_descending)):
#         log_error(
#             f"Error: Invalid data for a={alpha}, i={inf_chance}, r={rec_chance}"
#             )

#         return hysteresis_gaps

#     stable_t0 = find_stable_critical_t0(
#         inflist_ascending,
#         inflist_descending,
#         hys_threshold
#         )

#     if stable_t0 is not None:
#         stable_asc, stable_desc = stable_t0

#         if stable_asc is not None and stable_desc is not None:
#             hysteresis_gap = stable_asc - stable_desc
#             hysteresis_gaps.append(hysteresis_gap)

#     # if gaps:
#     #     ds_alpha["hysteresis_gaps"][i] = gaps[0]
#     # else:
#     #     ds_alpha["hysteresis_gaps"][i] = np.nan

#     return inflist_ascending, inflist_descending, hysteresis_gaps
