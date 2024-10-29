"""Module contains simulation logic."""
from complex_contagions_package.diffusion_behavior import recover, spread
from complex_contagions_package.network_generator import countinf, makeg


def simulate(
        t0, g_type, steps,
        infection_chance, recovery_chance,
        ascending=True, prev_g=None
        ):
    """Defines simulation logic according to diffusion behavior and alpha ratio.

    Parameters:
    - t0: threshold value of infection condition
    - alpha: ratio of infection rate to recovery rate
    - ascending: determines whether the simulation is ascending or descending
    - steps: number of simulation steps
    - prev_g: previous network state (used only for descending simulation)
    """
    simulation_type = "ascending" if ascending else "descending"
    g = makeg(t0, g_type, simulation_type=simulation_type, prev_g=prev_g)

    inflist = []

    for _ in range(steps):
        spread(g, infection_chance)
        recover(g, recovery_chance)
        inflist.append(countinf(g))

    return inflist, g

def run_hysteresis_simulation(
        g_type, steps,
        t0_values_ascending, t0_values_descending,
        inf_chance, rec_chance
        ):
    """Runs hysteresis simulation with ascending and descending T0 values.

    Simulation for descending t0 starts with the network state of the last simulation
    for ascending t0.

    Returns:
    - inflist_ascending: results of ascending simulation
    - inflist_descending: results of descending simulation
    """
    network_state_for_descending = None

    inflist_ascending = []

    for idx, t0 in enumerate(t0_values_ascending):
        inflist, g = simulate(t0, g_type, steps, inf_chance, rec_chance, ascending=True)
        inflist_ascending.append((idx, t0, inflist))

        if t0 == t0_values_ascending[-1]:
            network_state_for_descending = g.copy()

    inflist_descending = []

    for idx, t0 in enumerate(t0_values_descending):
        inflist, _ = simulate(t0, g_type, steps,
                              inf_chance, rec_chance,
                              ascending=False, prev_g=network_state_for_descending)
        inflist_descending.append((idx, t0, inflist))

    return inflist_ascending, inflist_descending
