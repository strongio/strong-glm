from typing import Optional

import numpy as np



def km_summary(time: np.ndarray,
               is_upper_cens: np.ndarray,
               lower_trunc: Optional[np.ndarray] = None) -> 'DataFrame':
    from pandas import DataFrame

    time = np.asanyarray(time)
    is_upper_cens = np.asanyarray(is_upper_cens)
    if lower_trunc is not None:
        lower_trunc = np.asanyarray(lower_trunc)

    if lower_trunc is not None and not np.isclose(lower_trunc, 0.0).all():
        raise NotImplementedError("`km_summary` not currently implemented if there's lower-truncation.")
    sorted_times = np.unique(time)

    df = {'time': [], 'num_at_risk': [], 'num_events': [], 'km_estimate': []}

    survival = 1.0
    num_at_risk = None
    for t in sorted_times:
        if num_at_risk is None:
            num_at_risk = len(time)
        is_time_bool = (t == time)
        num_censored = np.sum(is_upper_cens[is_time_bool]).item()
        num_events = np.sum(is_time_bool).item() - num_censored
        survival *= (1. - num_events / num_at_risk)

        df['time'].append(t.item())
        df['num_at_risk'].append(num_at_risk)
        df['num_events'].append(num_events)
        df['km_estimate'].append(survival)

        # for next iter:
        num_at_risk -= (num_events + num_censored)

    return DataFrame(df)
