
# 基于已有数据进行辅助指标的计算:
# 1. MACD指标的计算仅判断，DIF线向上突破MACD和向下突破MACD的情况
# 2. DMA指标判断DMA向上交叉平均线和向下交叉平均线时的情况，需要进一步确定short, long以及计算AMA时的window
# 3. TRIX为中长期指标


def calculate_factor(factor):
    """
    factor: str, the name of factor
    return: pd.DataFrame, values are dummy valriables of 1, 0 and -1
    Note:
    """

    factor = factor.upper() # Ignore the upper and lower case

    if factor == "MACD":
        """
        仅判断了DIF线向上突破MACD和向下突破MACD的情况
        """
        emas = cls.ewm(min_periods = 1, span = 12, ignore_na = True , adjust = False).mean()
        emal = cls.ewm(min_periods = 1, span = 26, ignore_na = True , adjust = False).mean()
        dif = emas - emal
        dea = dif.ewm(min_periods = 1, span = 9, ignore_na = True, adjust = False).mean()
        macd = (dif - dea) * 2
        result = ((dif > 0) & (macd > 0) & (dif.shift(1) < macd) & (dif > macd)) * 1 + ((dif > 0) & (macd > 0) & (dif.shift(1) > macd) & (dif < macd)) * -1

    if factor == "DMA":
        """
        需要进一步确认short和long的值以及计算ama的时候的window选择
        """
        dma = cls.rolling(window = short, min_periods = 1).mean() - cls.rolling(window = long, min_periods = 1).mean()
        ama = dma.rolling(window = short, min_periods = 1).mean()
        result = ((ama.shift(1) < dma) & (ama > dma)) * 1 + ((ama.shift(1) > dma) &(ama < dma)) * -1
        return result


    if factor == "TRIX":
        """
        中长期指标，需要对期限进行进一步的限定，目前设置第一次指数移动平均的window为12,计算MATRIX的window为20
        """
        n = 12
        m = 20
        ax = cls.ewm(span = n, min_periods = 1, ignore_na = True, adjust = False).mean()
        bx = ax.ewm(span = n, min_periods = 1, ignore_na = True, adjust = False).mean()
        tr = bx.ewm(span = n, min_periods = 1, ignore_na = True, adjust = False).mean()
        trix = (tr - tr.shift(1)) / (tr.shift(1)) * 100
        matrix = trix.rolling(window = m).mean()
