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



def equally_weighted_method(fac_lst):
    """
    Compose factors using equally_weighted_method

    fac_lst -> list: the name of factors need composing

    return -> DataFrame: result_compose
    """
    example_name = fac_lst[0]
    example_data = pd.read_csv("/home/Data/data_decay/{}_decay.csv".format(example_name),index_col = 0)

    res = example_data.copy() * 0
    for each in fac_lst:
        data = pd.read_csv("/home/Data/data_decay/{}_decay.csv".format(each), index_col = 0)
        res += data

    return res




def rolling_ols_method(fac_lst, ret, direction_dic, train_begin, train_end, test_begin, test_end):
    """
    Compose factors using regression year by year, need to specify the train period and test period

    fac_lst -> list: the name of factors need composing
    ret: -> DataFrame: daily return of every stock, ui2.shift(-1) / ui2 -1
    direction_dic -> dictionary: direction dictionary of each factor
    train_begin -> int: begin train date
    train_end -> int: end train date
    test_begin -> int: begin test date
    test_end -> int: end test date
    """

    train_df = pd.DataFrame()
    for each in fac_lst:
        data = pd.read_csv("/home/Data/data_decay/{}_decay.csv".format(each),index_col = 0)
        result_train = data.loc[train_begin, train_end]
        result_train = np.concatenate(np.array(result_train.T),axis = 0)
        train_df[each] = result_train
    train_df = train_df.replace(0,np.nan)
    train_df["y"] = np.concatenate(np.array(ret.loc[train_begin,train_end].T),axis = 0)
    train_df = train_df.dropna(thresh = 2).replace(np.nan,0)
    model = linear_model.LinearRegreesion()
    train_df_x = train_df.iloc[:,:-1]
    assert "y" not in train_df_x.columns
    model.fit(train_df_x, train_df.y)
    coef = pd.Series(model.coef_)

    res = data.loc[test_begin:test_end] * 0
    count = 0
    for ind, each in enumerate(fac_lst):
        if (np.sign(coef[ind]) * direction_dic[each]) == 1:
            count += 1
            data = pd.read_csv("/home/Data/data_decay/{}_decay.csv".format(each),index_col = 0).loc[test_begin,test_end]
            res -= coef[ind] * data

    print("According to our prior directions, we have %d qualified factors" %count)

    return res




def rolling_rf(fac_lst, train_begin, train_end, test_begin, test_end):
    """
    Train random forest model and use the predict value as compose factor of our factors

    fac_lst -> list: the name of factors need composing
    train_begin -> int: begin train date
    train_end -> int: end train date
    test_begin -> int: begin test date
    test_end -> int: end test date
    """
    train_df = pd.DataFrame()
    for i in range(len(fac_lst)):
        data = res_lst[i]
        result_train = data.loc[train_begin:train_end]
        result_train = np.concatenate(np.array(result_train.T),axis = 0)
        train_df[fac_lst[i]] = result_train
    train_df = train_df.replace(0, np.nan)
    train_df["y"] = np.concatenate(np.array(ret.loc[train_begin:train_end].T),axis = 0)
    train_df = train_df.dropna(thresh = 2).replace(np.nan,0)
    reg = RandomForestRegressor()
    train_df_x = train_df.iloc[:,:-1]
    assert "y" not in train_df_x.columns
    reg.fit(train_df_x, train_df.y)
    test_df = pd.DataFrame()
    for i in range(len(fac_lst)):
        result_test = res_lst[i].loc[test_begin:test_end]
        result_test = np.concatenate(np.array((result_test.T),axis = 0))
        test_df[fac_lst[i]] = result_test
    res_value = np.array(reg.predict(test_df))
    ind,col = data.loc[test_begin:test_end].index, data.loc[test_begin:test_end].columns
    length = len(ind)
    res = pd.DataFrame(res_values.reshape[-1,length].T), index = ind, columns = col)
    return -res



def rolling_perday(fac_lst, ret, res_dic, direction_dic, days, rolling_window = 500):
    """
    Use last rolling_window days data to train linear model and get factors' weights. Compose the next day's factors based on the weights

    fac_lst -> list: the name of factors need composing
    ret -> DataFrame: daily return of each stock
    res_dic -> dictionary of DataFrame : each factor's result(decay or not decay)
    direction_dic -> dictionary: each factor's direction
    days -> int: the number of days you need
    rolling_window -> int: the number of days you use to trian the model
    """
    result = res_dic[fac_lst[0]].iloc[rolling_window + 1:]
    ind_all = result.index.tolist()

    if days == -1:
        pass
    elif days <= len(ind_all):
        ind_all = ind_all[:days]
    elif days > len(ind_all):
        print("Error, variable days is out of range, function ends")
        return

    for ind in ind_all:
        train_df = pd.DataFrame()
        pt = res_dic[fac_lst[0]].index.get_loc(ind)

        for each in fac_lst:
            result_train = res_dic[each].iloc[pt - (rolling_window + 1):pt - 1]
            result_train = np.concatenate(np.array(result_train.T),axis = 0)
            train_df[each] = result_train
        train_df = train_df.replace(0, np.nan)
        train_df["y"] = np.concatenate(np.array(ret.iloc[pt - (rolling_window + 1):pt - 1].T),axis = 0)
        train_df = train_df.dropna(thresh = 2).replace(np.nan,0)
        model = linear_model.LinearRegreesion()
        train_df_x = train_df.iloc[:,:-1]
        assert "y" not in train_df_x.columns,"explained variable in explaining variables"
        model.fit(train_df_x, train_df.y)
        coef = pd.Series(model.coef_)

        for k, each in enumerate(fac_lst):
            if np.sign(coef[k]) * direction_dic[each] == 1:
                result.loc[ind] -= res_dic[each].loc[ind] * coef[k]

    return result.loc[ind_all]
