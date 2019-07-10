"""
用哑变量将形态因子进行量化表达视为事件因子，买入信号发生为1，卖出信号发生为-1，没有信号发生为0
"""

def calculate_factor(factor):
    """
    factor -> str: the name of factor
    return -> DataFrame: values are dummy valriables of 1, 0 and -1

    Note: 1 is the signal of buying and -1 is the signal of selling
    """

    factor = factor.upper() # Ignore the upper and lower case

    if factor == "MACD":
        """
        平滑异同平均指标

        To decide:
        仅判断了DIF线向上突破MACD和向下突破MACD的情况
        忽略了研报中描述意义不明确的情况
        """
        emas = cls.ewm(min_periods = 1, span = 12, ignore_na = True , adjust = False).mean()
        emal = cls.ewm(min_periods = 1, span = 26, ignore_na = True , adjust = False).mean()
        dif = emas - emal
        dea = dif.ewm(min_periods = 1, span = 9, ignore_na = True, adjust = False).mean()
        macd = (dif - dea) * 2
        result = ((dif > 0) & (macd > 0) & (dif.shift(1) < macd.shift(1)) & (dif > macd)) * 1 \
                 + ((dif > 0) & (macd > 0) & (dif.shift(1) > macd.shift(1)) & (dif < macd)) * -1
                 + ((dif < 0) & (macd < 0) & (dif.shift(1) > macd.shift(1)) & (dif < macd)) * 1
        return result


    if factor == "DMA":
        """
        平行线差指标，中短期指标，dma为短期平均值减去长期平均值，ama为dma的平均值

        To decide:
        需要进一步确认short和long的值以及计算ama的时候的window选择
        """
        dma = cls.rolling(window = short, min_periods = 1).mean() - cls.rolling(window = long, min_periods = 1).mean()
        ama = dma.rolling(window = short, min_periods = 1).mean()
        result = ((dma.shift(1) < ama.shift(1)) & (dma > ama)) * 1 \
                 +((dma.shift(1) > ama.shift(1)) & (dma < ama))* -1
        return result


    if factor == "TRIX":
        """
        三重指数平滑移动平均，中长期指标，目前设置第一次指数移动平均的window为12,计算MATRIX的window为20

        To decide:
        进一步设置指数移动平均和简单平均的window
        """
        n = 12
        m = 20
        ax = cls.ewm(span = n, min_periods = 1, ignore_na = True, adjust = False).mean()
        bx = ax.ewm(span = n, min_periods = 1, ignore_na = True, adjust = False).mean()
        tr = bx.ewm(span = n, min_periods = 1, ignore_na = True, adjust = False).mean()
        trix = (tr - tr.shift(1)) / (tr.shift(1)) * 100
        matrix = trix.rolling(window = m).mean()
        result = ((trix.shift(1) < matrix.shift(1)) & (trix > matrix)) * 1 \
                 + ((trix.shift(1) > matrix) & (trix < matrix)) * -1
        return result

    if factor == "BBI":
        """
        多空指数，将不同日数移动平均线加权之后的综合指标，属于均线型指标，一般选用3日，6日，12日，24日等4条平均线

        To decide:
        各个均线的window的值
        高价区跌破BBI的定义，目前高价区是指过去三天的值比BBI高的区域
        低价区突破BBI的定义，目前低价区是指过去三天的值比BBI低的区域
        """
        ma3 = cls.rolling(window = 3, min_periods = 1).mean()
        ma6 = cls.rolling(windwo = 6, min_periods = 1).mean()
        ma12 = cls.rolling(window = 12, min_periods = 1).mean()
        ma24 = cls.rolling(window = 24, min_periods = 1).mean()
        bbi = (ma3 + ma6 + ma12 + ma24)/4
        result = ((cls.shift(3) < bbi.shift(3)) & (cls.shift(2) < bbi.shift(2)) & (cls.shift(1) < bbi.shift(1)) & (cls > bbi)) * 1 \
                 +((cls.shift(3) > bbi.shift(3)) & (cls.shift(2) > bbi.shift(2)) & (cls.shift(1) > bbi.shift(1)) & (cls < bbi))* -1
        return result


    if factor == "DDI":
        """
        方向标准离差指数，通过分析DDI柱状线进行判断
        tr: 最高价与昨日最高价的绝对值，和最低价与昨日最低价的绝对值中的大的数
        dmz: 如果（最高价 + 最低价） <= （昨日最高价 + 昨日最低价，则dmz为0；
             如果（最高价 + 最低价） > (昨日最高价 + 昨日最低价)，则dmz的值为tr的值
        dmf: 如果（最高价 + 最低价） >= （昨日最高价 + 昨日最低价，则dmf为0；
             如果（最高价 + 最低价） < (昨日最高价 + 昨日最低价)，则dmf的值为tr的值
        diz = N个周期的dmz的和/（N个周期DMZ的和 + N个周期DMF的和）
        dif = N个周期的dmf的和/（N个周期DMZ的和 + N个周期DMF的和)
        ddi = diz - dif

        To decide: 周期N的值，目前取值为20
        """
        n = 20

        tr = np.maximum(np.abs(high.diff(1)), np.abs(low.diff(1)))

        dmz = pd.DataFrame(index = tr.index, columns = tr.columns)
        dmz[(high + low) <= (high.shift(1) + low.shift(1))] = 0
        dmz[(high + low) > (high.shift(1) + low.shift(1))] = tr[(high + low) > (high.shift(1) + low.shift(1))]

        dmf = pd.DataFrame(index = tr.index, columns = tr.columns)
        dmf[(high + low) >= (high.shift(1) + low.shift(1))] = 0
        dmf[(high + low) < (high.shift(1) + low.shift(1))] = tr[(high + low) < (high.shift(1) + low.shift(1))]

        diz = dmz.rolling(window = n, min_periods = 1).sum()/(dmz.rolling(window = n, min_periods = 1).sum() + dmf.rolling(window = n, min_periods = 1).sum())
        dif = dmf.rolling(window = n, min_periods = 1).sum()/(dmz.rolling(window = n, min_periods = 1).sum() + dmf.rolling(window = n, min_periods = 1).sum())

        ddi = diz - dif

        result = ((ddi.shift(1) < 0) & (ddi > 0)) * 1\
                 + ((ddi.shift(1) > 0) & (ddi < 0)) * -1

        return result

    if factor == "DMI":
        """
        动向指标，一种中长期的股市分析指标
        pos_dm: 当日的最高价减去前一日的最高价，如果值为负数，则记为0，是一个非负的变量
        neg_dm: 前一日的最低价减去当日的最低价，如果直接为负数，则记为0, 是一个非负的变量
        在将pos_dm和neg_dm进行比较，值较大的继续保留，值较小的则归为0，这样保证了每天的动向就是正动向，负动向和无动向
        tr: 真实波幅：当日的最高价减去当日的最低价，当日的最高价减去前一日的收盘价，当日的最低价减去前一日的收盘价 三者中的数值的绝对值的最大值
        pos_di: 正方向线：(pos_dm/tr) *100，要用平滑移动平均的pos_dm, tr来计算
        neg_di: 负方向线：(neg_dm/tr) *100，要用平滑移动平均的neg_dm, tr来计算
        dx: 动向指数：np.abs(pos_di-neg_di）/ (pos_di + neg_di) * 100
        adx: dx的EMA，平滑移动平均算

        To decide:计算pos_di时移动平均的window，目前设为12
                  计算adx的时候移动平均的window，目前设为12
        """

        n = 12

        pos_dm = np.maximum(high.diff(1), 0)
        neg_dm = np.maximum(-low.diff(1), 0)

        pos_dm[pos_dm < neg_dm] = 0
        neg_dm[neg_dm <= pos_dm] = 0

        tr = np.maximum(np.maximum(np.abs(high - low), np.abs(high - cls.shift(1))), np.abs(low - cls.shift(1)))

        pos_di = pos_dm.ewm(span = n, min_periods = 1, ignore_na = True, adjust = False).mean()/tr.ewm(span = 12, min_periods = 1, adjust = False, ignore_na = True).mean() * 100
        pos_di = pos_di.replace(float("inf"),0)

        neg_di = neg_dm.ewm(span = n, min_periods = 1, ignore_na = True, adjust = False).mean()/tr.ewm(span = 12, min_periods = 1, adjust = False, ignore_na = True).mean() * 100
        neg_di = neg_di.replace(float("inf"),0)

        dx = np.abs(pos_di - neg_di)/(pos_di + neg_di) * 100

        m = 12
        adx = dx.ewm(span = m, min_periods = 1, ignore_na = True, adjust = False).mean()
