"""Module contains simulation logic."""
from complex_contagions_package.diffusion_behavior import recover, spread
from complex_contagions_package.logging import log_info
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
        inf_chance, rec_chance,
        network_states_ascending, network_states_descending
        ):
    """Runs hysteresis simulation with ascending and descending T0 values.

    Simulation for descending t0 starts with the network state of the corresponding
    ascending t0 simulation instead of just using the last t0's network state.

    Returns:
    - inflist_ascending: results of ascending simulation
    - inflist_descending: results of descending simulation
    """
    inflist_ascending = []

    for idx, t0 in enumerate(t0_values_ascending):
        if idx == 0:
            inflist, g = simulate(t0, g_type, steps, inf_chance, rec_chance, ascending=True)
            #log_info("Initial asc_network state created")
        else:
            prev_t0 = t0_values_ascending[0]#[idx-1]
            g = network_states_ascending.get(prev_t0)
            #log_info(f"Asc_network state loaded for {t0}")
            inflist, g = simulate(t0, g_type, steps, inf_chance, rec_chance, ascending=True, prev_g=g)

        inflist_ascending.append((idx, t0, inflist))
        network_states_ascending[t0] = g.copy()
        #log_info(f"Saved asc_network state for t0={t0}")

    inflist_descending = []

    for idx, t0 in enumerate(t0_values_descending):
        prev_t0 = t0_values_ascending[-1]
        g = network_states_ascending.get(prev_t0)
        inflist, _ = simulate(t0, g_type, steps, inf_chance, rec_chance, ascending=False, prev_g=g)
        # if idx == 0:
        #     prev_t0 = t0_values_ascending[-1]
        #     g = network_states_ascending.get(prev_t0)
        #     #log_info("Initial desc_network state loaded")
        #     inflist, g = simulate(t0, g_type, steps, inf_chance, rec_chance, ascending=False, prev_g=g)
        # else:
        #     prev_t0 = t0_values_descending[idx-1]
        #     g = network_states_ascending.get(prev_t0)
        #     #log_info(f"Desc_network state loaded for {t0}")
        #     inflist, g = simulate(t0, g_type, steps, inf_chance, rec_chance, ascending=False, prev_g=g)  

        inflist_descending.append((idx, t0, inflist))
        #network_states_descending[t0] = g.copy
        #log_info(f"Saved desc_network state for t0={t0}")

    return inflist_ascending, inflist_descending

# def run_hysteresis_simulation(
#         g_type, steps,
#         t0_values_ascending, t0_values_descending,
#         inf_chance, rec_chance, network_states
#         ):
#     """Runs hysteresis simulation with ascending and descending T0 values.

#     Simulation for descending t0 starts with the network state of the corresponding
#     ascending t0 simulation instead of just using the last t0's network state.

#     Returns:
#     - inflist_ascending: results of ascending simulation
#     - inflist_descending: results of descending simulation
#     """
#     inflist_ascending = []

#     for idx, t0 in enumerate(t0_values_ascending):
#         inflist, g = simulate(t0, g_type, steps, inf_chance, rec_chance, ascending=True)
#         inflist_ascending.append((idx, t0, inflist))
#         network_states[t0] = g.copy()
#         #log_info(f"Saved network state for t0={t0}")

#     inflist_descending = []

#     for idx, t0 in enumerate(t0_values_descending):
#         #log_info(f"Trying to load network state for t0={t0}")
#         prev_g = network_states.get(t0)
#         if prev_g is None:
#             raise ValueError(f"Missing network state for t0={t0} in descending simulation.")
#         inflist, _ = simulate(t0, g_type, steps,
#                               inf_chance, rec_chance,
#                               ascending=False, prev_g=prev_g)
#         inflist_descending.append((idx, t0, inflist))

#     return inflist_ascending, inflist_descending
