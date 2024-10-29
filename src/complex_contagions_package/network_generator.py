"""Network generator."""
import random

import networkx as nx


def makeg(t0, g_type, simulation_type="ascending", prev_g=None):
    """Generates the network according to chosen type.

    Parameters:
    - t0: threshold value of infection chance
    - g_type: Graph (set in config)
    - simulation_type: determines whether the simulation runs for asc- or descending t0
    - prev_g: network state for last t0 in asc order (as starting point for desc t0)
    """
    if g_type is None:
        raise ValueError("g_type must be defined!")

    if simulation_type == "ascending":
        g = g_type.copy()
        nx.set_node_attributes(g, t0, "t")
        nx.set_node_attributes(g, 0, "inf")

        # Random node + all its neighbors get infected:
        r = random.choice(list(g))
        g.nodes[r]["inf"] = 1
        for n in g[r]:
            g.nodes[n]["inf"] = 1

    elif simulation_type == "descending":
        if prev_g is None:
            raise ValueError("prev_g must be provided for ascending simulation.")

        g = prev_g.copy()

        #nx.set_node_attributes(g, t0, "t")
        #nx.set_node_attributes(g, 0, "inf")

    return g

def countinf(g):
    """Counts the number of infections.

    Parameters:
    - G: Graph
    """
    allinf = 0
    for n in g:
        allinf += g.nodes[n]["inf"]
    return allinf
