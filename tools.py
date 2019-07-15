"""
Some auxiliary functions of factors
"""

def check_signals(data):
    """
    To find if signals appear singlely, or happens 2 in a row even, 3 in a row

    data -> DataFrame: single factor results

    return -> dictionary: count number of single signals, two signals in a row and three signals in a row
    """

    one = ((res == 1) & (res.shift(-1) !=1 )).sum().sum()
    two = ((res == 1) & (res.shift(-1) == 1) & (res.shift(-2) != 1)).sum().sum()
    three = ((res == 1) & (res.shift(-1) == 1) & (res.shift(-2) == 1) & (res.shift(-3) != 1)).sum().sum()

    total = data.sum().sum()

    dic = {}
    dic["single"] = (one, one / total)
    dic["two_in_a_row"] = (two, two / total)
    dic["three_in_a_row"] = (three, three / total)

    return dic


def backtest_v2(data, period):
    """
    To backtest the result of one signal, focus on relative return:
    - 1 and 1's complementary;
    - -1 and -1's complementary;
    - 1 and -1

    data -> DataFrame: if signals happens or not, values of 1, -1 and 0
    period -> int: when the signal happens, our holding period

    return -> 1 and 1's complementary， -1 and -1's complementary, 1 and -1 信号的information rate, number of 1's signals and number of -1's signals
    """

    # Update and filter the result using Stock_Pool
    result = data.copy() * 0
    result[(data == 1) & (Stock_Pool == 1)] = 1
    result[(data == -1) & (Stock_Pool == 1)] = -1

    pos_num = (result == 1).sum().sum()
    neg_num = (result == -1).sum().sum()

    def calculate_weight(result):
        """
        For each day, calculate the equal weight of stocks that have signals

        result -> DataFrame: the result of factor

        return -> DataFrame: the weight
        """
        weight_sum = np.sum(result, axis = 1).replace(0,1)
        return result.div(weight_sum, axis = "index")

    long_weight = calculate_weight((result == 1) * 1)
    op_long_weight = calculate_weight((result != 1) * 1)

    short_weight = calculate_weight((result == -1) * 1)
    op_short_weight = calculate_weight((result != -1) * 1)

    period_ret = ((ui2.shift(-period)/ui2).pow(1/period) - 1).fillna(0)

    long_ret = np.sum(long_weight * period_ret, axis = 1) #supposed to be positive values
    op_long_ret = np.sum(op_long_weight * period_ret, axis = 1)
    op_long_ret = op_long_ret.where((result == 1).sum(axis = 1).astype(bool)).fillna(0)



    short_ret = np.sum(short_weight * period_ret, axis = 1) # Supposed to be negative values
    op_short_ret = np.sum(op_short_weight * period_ret, axis = 1)
    op_short_ret = op_short_ret.where((result == -1).sum(axis = 1).astype(bool)).fillna(0)



    ret_long_opt = long_ret - op_long_ret
    ret_opt_short = op_short_ret - short_ret

    ret_long_short = long_ret.where((result == -1).sum(axis = 1).astype(bool)).fillna(0) - short_ret.where((result == 1).sum(axis = 1).astype(bool)).fillna(0)

    asset_l = [1]
    asset_s = [1]
    asset_ls = [1]

    for i in long_weight.index:
        asset_l.append(asset_l[-1] * (ret_long_opt[i] + 1))
        asset_s.append(asset_s[-1] * (ret_opt_short[i] + 1))
        asset_ls.append(asset_ls[-1] * (ret_long_short[i] + 1))

    ir_l = (pow(asset_l[-1], 1/9) - 1) / (np.std(ret_long_opt) * np.sqrt(250))
    ir_s = (pow(asset_s[-1], 1/9) - 1) / (np.std(ret_opt_short) * np.sqrt(250))
    ir_ls = (pow(asset_ls[-1], 1/9) - 1) / (np.std(ret_long_short) * np.sqrt(250))

    dt = pd.to_datetime(cls.index, format = "%Y%m%d").tolist() + [datetime.datetime.striptime(str(20181115), "%Y%m%d")] # make time stamp and make dimensions match
    fig = plt.figure(figsize = [12,8])
    ax1 = fig.add_subplot(111)
    ax1.plot(dt,pd.Series(asset_l), label = "long - opt", color = "red")
    ax1.plot(dt,pd.Series(asset_s), label = "opt - short", color = "blue")
    ax1.plot(dt,pd.Series(asset_ls), label = "long - short", color = "green")
    ax1.legend(loc = "upper right", fontsize = 10)

    plt.show()
    print("The IR of long-opt is %f\n\
    The IR of short-opt is %f\n\
    The IR of long-short is\n\
    The number of positive signals are %d\n\
    The number of negative signals are %d" %(ir_l, ir_s, ir_ls, pos_num, neg_num))

    return ir_l, ir_s, ir_ls, pos_num, neg_num
