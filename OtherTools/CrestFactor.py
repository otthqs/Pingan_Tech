def calculate_crestfactor(cls, begin_date, rolling_window = 10, thresh = 0.1):
    """
    1. 波峰定义：收盘价cls,在一个rolling_window中，当前的值比上一个window的最小值大一个threshold，如10%，比下一个window中的最小值大一个threshold,
    并且是局部极大值（比前一天和后一天的收盘价高）
    2. Note: 在这个标准下，判断一个cls是否是波峰，至少需要当期之前有至少一天的数据，在当期之后有一期的数据。但是这个判断是动态的：
    收盘价为 10 12 11 9。站在第三天看第二天，就不是波峰。但是站在第四天看第二天就是波峰；
    但是当期之前有完整的一个rolling_window，在当期之后有一个完整的rolling_window的话，那么是否为波峰就是在全局都是准确的，随着时间的推移是不会变的；
    此函数维计算的因子值是：离上一个波峰出现到现在的天数

    cls -> DataFrame: close price
    begin_date -> int: begin_date is in the format of int
    rolling_window -> int: the rolling_window of deciding if one crest happens
    thresh -> float: the range of price increasing to decide if one crest happens
    """

    cls = cls.loc[begin_date:]

    #初始化要返回的结果，初始值全部为nan，当rolling_window = 10时，就从第22天开始有值（保证第11天波峰因子值不随时间的变化而变化）
    distance_factor = (cls.iloc[2 * rolling_window +1 : ] * 0).replace(0,np.nan)
    ind_all = distance_factor.index

    factor_real = np.array([np.nan] * len(cls.columns)) #需要维护的全局都是准确的天数

    for ind in ind_all:
        pt = cls.index.get_loc(ind)
        cls_temp = cls.iloc[pt - 2 * rolling_window -1 : pt + 1] # 计算当天天数时，需要过去21天的数据，这样第11天的是否为波峰是全局准确的
        cls_temp_back_min = cls_temp.rolling(window = rolling_window, min_periods = 1).min() #从第11天开始，有完整的过去10天的数据
        signal_observation = (cls_temp.iloc[rolling_window : -1].fillna(0)) * 0 # 站在第22天，观察到的过去10天（从第11天到底22天之间的10天是否有波峰出现，这个值可能会随着天数推移变化）

        for each in signal_observation.index: # 因为从第11天之后，到第22天，没有一个完整的window，所以是否出现波峰要单独判断
            length = min(rolling_window, rolling_window + 1 - signal_observation.index(get_loc(each))) #判断不是一个完整的window的话，到目前为止有几天
            cls_temp_forward_min = cls_temp.rolling(window = length, min_periods = 1).min()
            signal_observation.loc[each] = (((cls_temp / cls_temp_back_min.shift(1)) > (1 + thresh)) & ((cls_temp / cls_temp_forward_min.shift(-1 * length)) > (1 + thresh)) & (cls_temp > cls_temp.shift(1)) & (cls_temp > cls_temp.shift(-1))).loc[each]

        signal_real = signal_observation.iloc[0].astype(bool)

        #更新factor_real的值，当信号发生则为0，不发生则+=1
        factor_real[signal_real] = 0
        factor_real[~signal_real] += 1

        #从factor_real开始更新到今天的天数值
        factor_temp = factor_real.copy()

        for i in range(1, len(signal_observation)):
            signal_temp = signal_observation.iloc[i].astype(bool)
            factor_temp[signal_temp] = 0
            factor_temp[~signal_temp] += 1
            distance_factor.loc[ind] = factor_temp

        return distance_factor
