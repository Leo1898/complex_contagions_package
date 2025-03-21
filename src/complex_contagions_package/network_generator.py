"""Network generator."""
import random

import networkx as nx

def initialize_infection(g, infection_value=1): #infectin_value = 1 statt 0 wenn komplett infiziert sein soll
    """Random node + all its neighbors get infected."""
    r = random.choice(list(g))
    g.nodes[r]["inf"] = infection_value
    for n in g[r]:
        g.nodes[n]["inf"] = infection_value

def makeg(t0, g_type, simulation_type="ascending", prev_g=None):
    """Generates the network according to chosen type.

    Parameters:
    - t0: threshold value of infection chance
    - g_type: Graph (set in config)
    - simulation_type: determines whether the simulation runs for asc- or descending t0
    - prev_g: network state for previous t0

    Returns:
    - A networkx Graph object with updated attributes.
    """
    if g_type is None:
        raise ValueError("g_type must be defined!")

    # Create or copy network
    if prev_g is None:
        g = g_type.copy()
        nx.set_node_attributes(g, 0, "inf")  # Standard: Alle nicht infiziert
        #nx.set_node_attributes(g, 1, "inf")
    else:
        g = prev_g.copy()

    # Setze Threshold-Attribute
    nx.set_node_attributes(g, t0, "t")
    # for n in g.nodes:
    #     g.nodes[n]["t"] = max(0, np.random.normal(loc=t0, scale=0.1 * t0))

    if simulation_type == "ascending" and prev_g is None:
        initialize_infection(g, infection_value=1)

    if simulation_type == "descending" and prev_g is None:
        raise ValueError("prev_g must be provided for descending simulation.")

    return g

# def makeg(t0, g_type, simulation_type="ascending", prev_g=None):
#     """Generates the network according to chosen type.

#     Parameters:
#     - t0: threshold value of infection chance
#     - g_type: Graph (set in config)
#     - simulation_type: determines whether the simulation runs for asc- or descending t0
#     - prev_g: network state for last t0 in asc order (as starting point for desc t0)
#     """
#     if g_type is None:
#         raise ValueError("g_type must be defined!")

#     if simulation_type == "ascending":
#         g = g_type.copy()
#         nx.set_node_attributes(g, t0, "t")
#         # for n in g.nodes:
#         #     g.nodes[n]["t"] = max(0, np.random.normal(loc=t0, scale=0.1 * t0))

#         nx.set_node_attributes(g, 0, "inf")
#         #nx.set_node_attributes(g, 1, "inf")

#         # # Random node + all its neighbors get infected:
#         r = random.choice(list(g))
#         g.nodes[r]["inf"] = 1
#         for n in g[r]:
#             g.nodes[n]["inf"] = 1

#         # # Random node + all its neighbors get infected:
#         # r = random.choice(list(g))
#         # g.nodes[r]["inf"] = 0
#         # for n in g[r]:
#         #     g.nodes[n]["inf"] = 0

#     elif simulation_type == "descending":
#         if prev_g is None:
#             raise ValueError("prev_g must be provided for ascending simulation.")

#         g = prev_g.copy()

#         nx.set_node_attributes(g, t0, "t")
#         # for n in g.nodes:
#         #     g.nodes[n]["t"] = max(0, np.random.normal(loc=t0, scale=0.1 * t0))

#     return g

def countinf(g):
    """Counts the number of infections.

    Parameters:
    - G: Graph
    """
    allinf = 0
    for n in g:
        allinf += g.nodes[n]["inf"]
    return allinf
