"""
将已有的形态因子按照一定的方式进行合成
"""

def decay_method(decay_std, data):
    """
    One signal happens, we assume the signal's effect would last for several days and decay exponentially.

    decay_std -> array
    data -> array
    return -> int
    """

    decay_std = decay_std[::-1]
    res = data * decay_std[0]
    for i in range(1,len(decay_std)):
        res += (data.shift(i) * decay_std[i]).fillna(0)

    return res


def exponential_decay(alpha, len):
    """
    Give decay rate and decay length, return exponentially decay weight

    alpha -> float
    len -> int
    return -> array
    """

    exp_decay = []
    for i in range(len+1):
        exp_decay.append(np.exp(-alpha * i))

    decay_std = [(x - min(exp_decay))/(max(exp_decay) - min(exp_decay)) for x in exp_decay]
    decay_std = decay_std[::-1]

    return decay_std[1:]


def ir_compose(res_dic, period_dic, direction_dic, rolling_period, ui2, Stock_Pool):
    """
    Using rolling historical ir value as weight to compose a new continuous factor

    res_dic -> dictionary of DataFrame : each factor's result
    period_dic -> dictionary: every factor's effective holding period
    direction_dic -> dictionary: factor's direction, buying or selling
    rolling_period -> int: rolling window length
    Stock_Pool -> DataFrame: effective stock at each day, taking limit state into account
    ui2 -> DataFrame: cumulative rate of return on traits
    """

    def backtest(result, period, ui2, direction):
        """
        compute relative return of a single factor

        result -> DataFrame: one factor's result
        period -> int: effective holding period
        ui2 -> DataFrame: cumulative rate of return on traits
        direction -> int: buying or selling

        return -> np.array: relative return of one factor on each day
        """

        
