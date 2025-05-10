"""Module contains simulation logic."""
#from complex_contagions_package.analyser import find_stable_critical_t0
from complex_contagions_package.analyser import analyze_clusters
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
    initial_g = g.copy()
    inflist = []

    for _ in range(steps):
        spread(g, infection_chance)
        recover(g, recovery_chance)
        inflist.append(countinf(g))

    return inflist, g, initial_g

def run_hysteresis_simulation(
        i, g_type, steps,
        t0_values_ascending=None, t0_values_descending=None,
        inf_chance=None, rec_chance=None,
        network_states_ascending=None, network_states_descending=None,
        #overridet0=None
        ):
    """Runs hysteresis simulation with ascending and descending T0 values.

    Simulation for descending t0 starts with the network state of the corresponding
    ascending t0 simulation instead of just using the last t0's network state.

    Returns:
    - inflist_ascending: results of ascending simulation
    - inflist_descending: results of descending simulation
    """
    inflist_ascending = []

    if t0_values_ascending is not None:
        for idx, t0 in enumerate(t0_values_ascending):
            if idx == 0:
                inflist, g_asc, initial_g  = simulate(t0, g_type, steps,
                                                      inf_chance, rec_chance,
                                                      ascending=True,
                                                      prev_g=None
                                                      )
                #log_info("Initial asc_network state created")
            else:
                # #pink
                # prev_t0 = t0_values_ascending[idx-1]
                # g_prev = network_states_ascending[i].get(prev_t0)

                #blau und gruen
                g_prev = initial_g

                #log_info(f"Asc_network state loaded for {t0}")
                inflist, g_asc, _= simulate(t0, g_type, steps,
                                            inf_chance, rec_chance,
                                            ascending=True,
                                            prev_g=g_prev)

            cluster_stats = analyze_clusters(g_asc)
            inflist_ascending.append((idx, t0, inflist, cluster_stats))
            network_states_ascending[i][t0] = g_asc
            #log_info(f"Saved asc_network state for t0={t0}")

    inflist_descending = []

    # if t0_values_descending is not None and overridet0 is None:
    #         raise ValueError("Startzustand (start_t0_override) für absteigende Simulation fehlt!")

    if t0_values_descending is not None:
        # blau
        # prev_t0 = t0_values_descending[0] # prev_t0 = start_t0_override
        # g_prev = network_states_ascending[i].get(prev_t0)
        # if g_prev is None:
        #     log_info(
        #         f"Kein Netzwerkzustand für t0 = {t0} gefunden (simulation {i})"
        #         )

        for idx, t0 in enumerate(t0_values_descending):
            # # pink
            # if idx == 0:
            #     prev_t0 = t0_values_descending[0]
            #     g_prev = network_states_ascending[i].get(prev_t0)
            #     if g_prev is None:
            #         log_info(
            #             f"Kein Netzwerkzustand für t0 = {t0} gefunden (simulation {i})"
            #             )
            #     inflist, g_desc, _ = simulate(t0, g_type, steps,
            #                                   inf_chance, rec_chance,
            #                                   ascending=False, prev_g=g_prev)
            # else:
            #     prev_t0 = t0_values_descending[idx-1]
            #     g_prev = network_states_descending[i].get(prev_t0)
            #     if g_prev is None:
            #         log_info(
            #             f"Kein Netzwerkzustand für t0 = {t0} gefunden (simulation {i})"
            #             )
            #     inflist, g_desc, _ = simulate(t0, g_type, steps,
            #                                   inf_chance, rec_chance,
            #                                   ascending=False, prev_g=g_prev)

            #gruen
            prev_t0 = t0_values_descending[idx]
            g_prev = network_states_ascending[i].get(prev_t0)
            if g_prev is None:
                log_info(
                    f"Kein Netzwerkzustand für t0 = {t0} gefunden (simulation {i})"
                    )
            #blau und gruen
            inflist, g_desc, _ = simulate(t0, g_type, steps,
                                     inf_chance, rec_chance,
                                     ascending=False,
                                     prev_g=g_prev)

            cluster_stats = analyze_clusters(g_desc)
            inflist_descending.append((idx, t0, inflist, cluster_stats))
            # pink
            #network_states_descending[i][t0] = g_desc
            #log_info(f"Saved desc_network state for t0={t0}")

    return inflist_ascending, inflist_descending


