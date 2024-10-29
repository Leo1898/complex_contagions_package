"""Simulation analysis."""
from complex_contagions_package.logging import log_error
from complex_contagions_package.simulator import run_hysteresis_simulation


def find_all_critical_t0(inflist_ascending, inflist_descending, hys_threshold):
    """Identifies critical t0 values.

    A t0 is defined to be critical when last step infected nodes for the previous t0
    differs by at least the amount of hys_threshold compared to the last step infected
    nodes of the current t0.

    Parameters:
    - inflist_ascending: infected nodes results for ascending t0 values
    - inflist_descending: infected nodes results for descending t0 values
    - hys_threshold: jump of end infections defined to be critical (set in Config)
    """
    critical_t0_values_asc = []
    previous_value_asc = inflist_ascending[0][2][-1]

    for (idx, t0, inflist) in inflist_ascending:
        current_value = inflist[-1]
        if abs(current_value - previous_value_asc) > hys_threshold:
            critical_t0_values_asc.append((idx, t0, current_value))
        previous_value_asc = current_value

    critical_t0_values_desc = []
    previous_value_desc = inflist_descending[0][2][-1]

    for (idx, t0, inflist) in inflist_descending:
        current_value = inflist[-1]  # Use the last value of the list
        if abs(current_value - previous_value_desc) > hys_threshold:
            critical_t0_values_desc.append((idx, t0, current_value))
        previous_value_desc = current_value

    return critical_t0_values_asc, critical_t0_values_desc

def is_stable_within_window(infectionlist, index, hys_threshold, window_size = 3):
    """Check if the infection rates within a sliding window are stable.

    Ensures the window is within the bounds of the infectionlist, collects values within
    the window and checks if all values in the window are within the hys_threshold.

    Parameters:
    - infectionlist: infected nodes results for asc/desc t0 values
    - index: current index
    - hys_threshold: jump of end infections defined to be critical (set in config)
    """
    end_index = min(index + window_size, len(infectionlist))

    window_values = [infectionlist[i][2][-1] for i in range(index, end_index)]

    return all(abs(window_values[i] - window_values[i-1]) < hys_threshold
               for i in range(1, len(window_values)))

def find_stable_critical_t0(inflist_ascending, inflist_descending, hys_threshold):
    """Find stable critical points for ascending and descending t0 values.

    Gets critical t0 values for ascending and descending order. Returns None if
    no critical values were found or returns both stable points if stable values
    within window were found (or last critical jump if none is stable).

    Parameters:
    - inflist_ascending: infected nodes results for ascending t0 values
    - inflist_descending: infected nodes results for descending t0 values
    - hys_threshold: jump of end infections defined to be critical (set in config)
    """
    critical_t0_values_asc, critical_t0_values_desc = find_all_critical_t0(
        inflist_ascending,
        inflist_descending,
        hys_threshold
        )

    if not critical_t0_values_asc and not critical_t0_values_desc:
        return None

    stable_asc = None
    for (idx, t0, _) in reversed(critical_t0_values_asc):
        if is_stable_within_window(inflist_ascending, idx, hys_threshold):
            stable_asc = t0
            break

    stable_desc = None
    for (idx, t0, _) in reversed(critical_t0_values_desc):
        if is_stable_within_window(inflist_descending, idx, hys_threshold):
            stable_desc = t0
            break

    return stable_asc or (critical_t0_values_asc[-1][1] if
                          critical_t0_values_asc else None), \
           stable_desc or (critical_t0_values_desc[-1][1] if
                           critical_t0_values_desc else None)

def calculate_hysteresis_gaps(
        hys_threshold, alpha, g_type,
        t0_values_ascending, t0_values_descending,
        inf_chance, rec_chance, steps):
    """Identifies hysteresis if present.

    Runs a simulation, checks if valid values are returned, calls stable critical values
    and calculates and returns the differnce between stable critical values as well as
    infection results of the simulation.

    Parameters:
    - inflist_ascending: infected nodes results for ascending t0 values
    - inflist_descending: infected nodes results for descending t0 values
    - hys_threshold: jump of end infections defined to be critical (set in Config)
    """
    hysteresis_gaps = []

    inflist_ascending, inflist_descending = run_hysteresis_simulation(
        g_type, steps,
        t0_values_ascending, t0_values_descending,
        inf_chance, rec_chance,
        )

    if any(x is None or not x for x in (inflist_ascending, inflist_descending)):
        log_error(
            f"Error: Invalid data for a={alpha}, i={inf_chance}, r={rec_chance}"
            )

        return hysteresis_gaps

    stable_t0 = find_stable_critical_t0(
        inflist_ascending,
        inflist_descending,
        hys_threshold
        )

    if stable_t0 is not None:
        stable_asc, stable_desc = stable_t0

        if stable_asc is not None and stable_desc is not None:
            hysteresis_gap = stable_asc - stable_desc
            hysteresis_gaps.append(hysteresis_gap)

    return inflist_ascending, inflist_descending, hysteresis_gaps
