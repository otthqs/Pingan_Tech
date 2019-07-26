"""
将已有的形态因子按照一定的方式进行合成
"""

def decay_method(decay_std, data):
    """
    One signal happens, we assume the signal's effect would last for several days and decay exponentially.

    Parm:
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

    Parm:
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
    Using rolling historical ir value as weight to compose a new continuous factor, decay period is the same as holding period

    Parm:
    res_dic -> dictionary of DataFrame : each factor's result
    period_dic -> dictionary: every factor's effective holding period
    direction_dic -> dictionary: factor's direction, buying or selling
    rolling_period -> int: rolling window length
    Stock_Pool -> DataFrame: effective stock at each day, taking limit state into account
    ui2 -> DataFrame: cumulative rate of return on traits

    return -> DataFrame: compose factor
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

        op_result = 1 - result
        result = ((result == 1) & (Stock_Pool == 1)) * 1
        op_result = ((op_result == 1) & (Stock_Pool == 1)) * 1

        def calculate_weight(result):
            """
            For each day, calculate the equal weight of stocks that have signals

            result -> DataFrame: the result of factor

            return -> DataFrame: the weight
            """

            weight_sum = np.sum(result, axis = 1).replace(0,1)
            return result.div(weight_sum, axis = "index")

        weight = calculate_weight(result)
        op_weight = calculate_weight(op_result)

        period_ret = ((ui2.shift(-period) / ui2).pow(1/period) -1).fillna(0)
        avg_ret = direction * np.sum(weight * period_ret, axis = 1)
        op_ret = direction * np.sum(op_weight * period_ret, axis = 1)

        avg_ret = avg_ret.replace(0, np.nan)
        relevant_ret = avg_ret - op_ret

        return relevant_ret

    ind = list(res_dic.values())[0].index
    k_ = list(res_dic.keys())[0]

    rel_return = pd.DataFrame(index = ind)
    for k,v in res_dic.items():
        rel_return[k] = backtest(v, period_dic[k], ui2, direction_dic[k])

    ir = rel_return.rolling(window = rolling_period, min_periods = 1).mean() / rel_return.rolling(window = rolling_period, min_periods = 1).std()
    weight_df = ir.iloc[rolling_period - 1:]

    weight_df_shift = weight_df.copy() * 0

    for each in weight_df_shift.columns:
        weight_df_shift[each] = weight_df[each].shift(period_dic[each])

    weight_df_shift = weight_df_shift.iloc[20:]
    weight_df_shift = np.maximum(weight_df_shift, 0)
    ind_ = weight_df_shift.index

    res_dic_ = {}
    for k,v in res_dic.items():
        decay_std = exponential_decay(1,period_dic[k])
        res_dic_[k] = decay_method(decay_std, v).loc[ind_]

    ir_compose = res_dic_[k_] * 0
    for k,v in res_dic_.items():
        if direction_dic[k] == 1:
            ir_compose -= v.mul(weight_df_shift[k], axis = 0)

        if direction_dic[k] == -1:
            if_compose += v.mul(weight_df_shift[k], axis = 0)

    return ir_compose



def ir_compose_v2(rel_dic, res_dic, period_dic, direction_dic, rolling_period, ui2, Stock_Pool):
    """
    Using rolling historical ir value as weight to compose a new continuous factor, decay period is the same as holding period
    Using rel_dic to replace the backtest process, increasing efficiency

    Parm:
    rel_dic -> dictionary of DataFrame : each factor's relative return
    res_dic -> dictionary of DataFrame : each factor's result
    period_dic -> dictionary: every factor's effective holding period
    direction_dic -> dictionary: factor's direction, buying or selling
    rolling_period -> int: rolling window length
    Stock_Pool -> DataFrame: effective stock at each day, taking limit state into account
    ui2 -> DataFrame: cumulative rate of return on traits

    return -> DataFrame: compose factor
    """


    ind = list(res_dic.values())[0].index
    k_ = list(res_dic.keys())[0]


    ir = rel_return.rolling(window = rolling_period, min_periods = 1).mean() / rel_return.rolling(window = rolling_period, min_periods = 1).std()
    weight_df = ir.iloc[rolling_period - 1:]

    weight_df_shift = weight_df.copy() * 0

    for each in weight_df_shift.columns:
        weight_df_shift[each] = weight_df[each].shift(period_dic[each])

    weight_df_shift = weight_df_shift.iloc[20:]
    weight_df_shift = np.maximum(weight_df_shift, 0)
    ind_ = weight_df_shift.index

    res_dic_ = {}
    for k,v in res_dic.items():
        decay_std = exponential_decay(1,period_dic[k])
        res_dic_[k] = decay_method(decay_std, v).loc[ind_]

    ir_compose = res_dic_[k_] * 0
    for k,v in res_dic_.items():
        if direction_dic[k] == 1:
            ir_compose -= v.mul(weight_df_shift[k], axis = 0)

        if direction_dic[k] == -1:
            if_compose += v.mul(weight_df_shift[k], axis = 0)

    return ir_compose



def ir_compose_v3(rel_return_input, fac_lst, period_dic, direction_dic, rolling_period, ui2, Stock_Pool):
        """
        Using rolling historical ir value as weight to compose a new continuous factor, decay period is the same as holding period
        Using res_dic_decay result to replace the decay process, increasing efficiency

        Parm:
        rel_return_input -> DataFrame : each factor's relative return
        fac_lst -> list: the list of selected factors in the compose process
        period_dic -> dictionary: every factor's effective holding period
        direction_dic -> dictionary: factor's direction, buying or selling
        rolling_period -> int: rolling window length
        Stock_Pool -> DataFrame: effective stock at each day, taking limit state into account
        ui2 -> DataFrame: cumulative rate of return on traits

        return -> DataFrame: compose factor
        """

        rel_return = pd.DataFrame()

        for each in fac_lst:
            rel_return[each] = rel_return_input[each]

        ir = rel_return.rolling(window = rolling_period, min_periods = 1).mean() / rel_return.rolling(window = rolling_period, min_periods = 1).std

        weight_df = ir.iloc[rolling_period-1:]
        weight_df_shift = weight_df.copy() * 0

        for each in weight_df_shift.columns:
            weight_df_shift[each] = weight_df[each].sbift(period_dic[each])
        weight_df_shift = weight_df_shift.iloc[20:]
        weight_df_shift = np.maximum(weight_df_shift,0)

        ind_ = weight_df_shift.index

        example = fac_lst[0]
        example_data = pd.read_csv("/home/Data/data_decay/{}_decay.csv".format(example),index_col = 0)

        ir_compose = example_data.loc[ind_] * 0

        for k in fac_lst:
            v = pd.read_csv("/home/Data/data_decay/{}_decay.csv".format(example),index_col = 0)

            if direction_dic[k] == 1:
                ir_compose -= v.mul(weight_df_shift[k], axis = 0)

            if direction_dic[k] == -1:
                ir_compose += v.mul(weight_df_shift[k], axis = 0)

        return ir_compose
