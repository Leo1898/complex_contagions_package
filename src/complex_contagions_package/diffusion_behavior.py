"""Set diffusion behavior."""
import numpy as np


def infect(g, n, inf_chance):
    """Defines infection behavior according to infection conditions.

    A node is to be infected if the ratio of infected to all neighbours is equal or
    above a certain threshold.

    Parameters:
    - g: Graph
    - n: nodes of the graph
    - inf_chance: probability that a node is infected when conditions are met
    """
    inf_neibs_ratio = inf_neibs(g, n)
    threshold = g.nodes[n]["t"]

    inf = False
    if inf_neibs_ratio >= threshold and np.random.random() < inf_chance:
        inf = True

    return inf

def spread(g, inf_chance):
    """Determines infection spreading (infected nodes) for each timestep.

    Parameters:
    - g: Graph
    - inf_chance: probability that a node is infected when conditions are met
    """
    newinf=[]
    #loop over all nodes
    for n in g:
        if(infect(g, n, inf_chance)):
            newinf.append(n)

    for x in newinf:
        g.nodes[x]["inf"] = 1

def recover(g, rec_chance):
    """Calculates recovered nodes for each timestep according to recovery probability.

    Parameters:
    - G: Graph
    - rec_chance: chance of recovery (set in Config)
    """
    for node in g:
        if np.random.random() < rec_chance:
            g.nodes[node]["inf"] = 0

def inf_neibs(g,n):
    """Returns the ratio of infected to all neighbours for each node and time step.

    Parameters:
    - g: Graph
    - n: nodes of the graph
    """
    inf = 0
    neibs=0
    for neib in g[n]:
        neibs+=1
        if g.nodes[neib]["inf"]==1:
            inf+=1

    return inf/neibs if neibs > 0 else 0
