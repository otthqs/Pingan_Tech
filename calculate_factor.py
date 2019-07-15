"""
用哑变量将形态因子进行量化表达视为事件因子，买入信号发生为1，卖出信号发生为-1，没有信号发生为0
需要用到的数据:开盘价，收盘价，最高价，最低价，成交量，成交额，股票池，限制状态，特质收益率
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
        短期的周期长度n, 长期的周期长度m, 计算dea的周期长度l
        """
        n = 12
        m = 26
        l = 9

        emas = cls.ewm(min_periods = 1, span = n, ignore_na = True , adjust = False).mean()
        emal = cls.ewm(min_periods = 1, span = m, ignore_na = True , adjust = False).mean()
        dif = emas - emal
        dea = dif.ewm(min_periods = 1, span = l, ignore_na = True, adjust = False).mean()
        macd = (dif - dea) * 2
        result = ((dif > 0) & (macd > 0) & (dif.shift(1) < macd.shift(1)) & (dif >= macd)) * 1\
                 + ((dif > 0) & (macd > 0) & (dif.shift(1) > macd.shift(1)) & (dif <= macd)) * -1\
                 + ((dif < 0) & (macd < 0) & (dif.shift(1) > macd.shift(1)) & (dif <= macd)) * 1


    elif factor == "DMA":
        """
        平行线差指标，中短期指标，dma为短期平均值减去长期平均值，ama为dma的平均值

        To decide:
        需要进一步确认n和m的值以及计算ama的时候的window选择
        """
        n = 10
        m = 50
        l = 10
        dma = cls.rolling(window = n, min_periods = n-2).mean() - cls.rolling(window = m, min_periods = m-n).mean()
        ama = dma.rolling(window = l, min_periods = l-2).mean()
        result = ((dma.shift(1) < ama.shift(1)) & (dma >= ama)) * 1 \
                 +((dma.shift(1) > ama.shift(1)) & (dma <= ama))* -1


    elif factor == "TRIX":
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
        matrix = trix.rolling(window = m, min_periods = m-5).mean()
        result = ((trix.shift(1) < matrix.shift(1)) & (trix >= matrix)) * 1 \
                 + ((trix.shift(1) > matrix) & (trix <= matrix)) * -1


    elif factor == "BBI":
        """
        多空指数，将不同日数移动平均线加权之后的综合指标，属于均线型指标，一般选用3日，6日，12日，24日等4条平均线

        To decide:
        各个均线的window的值n,m,l,h
        高价区跌破BBI的定义，目前高价区是指收盘价的价格比过去400天交易日收盘价Q3的值高的区域
        低价区突破BBI的定义，目前低价区是指收盘价的价格比过去400天交易日收盘价Q1的值低的区域
        """
        n = 3
        m = 6
        l = 12
        h = 24
        ma3 = cls.rolling(window = n, min_periods = 1).mean()
        ma6 = cls.rolling(windwo = m, min_periods = m-1).mean()
        ma12 = cls.rolling(window = l, min_periods = l-5).mean()
        ma24 = cls.rolling(window = h, min_periods = h-5).mean()
        bbi = (ma3 + ma6 + ma12 + ma24)/4
        q3 = cls.rolling(window = 400, min_periods = 200).quantile(0.75)
        q1 = cls.rolling(window = 400, min_periods = 200).quantile(0.25)

        result = ((cls < q1) & (cls.shift(1) < bbi.shift(1)) & (cls >= bbi)) * 1\
                +((cls > q3) & (cls.shift(1) > bbi.shift(1)) & (cls <= bbi)) * -1



    elif factor == "DDI":
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

        To decide:
        周期N的值，目前取值为20
        """
        n = 20

        tr = np.maximum(np.abs(high.diff(1)), np.abs(low.diff(1)))

        dmz = pd.DataFrame(index = tr.index, columns = tr.columns)
        dmz[(high + low) <= (high.shift(1) + low.shift(1))] = 0
        dmz[(high + low) > (high.shift(1) + low.shift(1))] = tr[(high + low) > (high.shift(1) + low.shift(1))]

        dmf = pd.DataFrame(index = tr.index, columns = tr.columns)
        dmf[(high + low) >= (high.shift(1) + low.shift(1))] = 0
        dmf[(high + low) < (high.shift(1) + low.shift(1))] = tr[(high + low) < (high.shift(1) + low.shift(1))]

        diz = dmz.rolling(window = n, min_periods = n-2).sum()/(dmz.rolling(window = n, min_periods = n-2).sum() + dmf.rolling(window = n, min_periods = n-2).sum())
        dif = dmf.rolling(window = n, min_periods = n-2).sum()/(dmz.rolling(window = n, min_periods = n-2).sum() + dmf.rolling(window = n, min_periods = n-2).sum())

        ddi = diz - dif

        result = ((ddi.shift(1) < 0) & (ddi >= 0)) * 1\
                 + ((ddi.shift(1) > 0) & (ddi <= 0)) * -1



    elif factor == "DMI":
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

        To decide:
        计算pos_di时移动平均的window，目前设为14
        计算adx的时候移动平均的window，目前设为6
        """

        n = 14

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

        m = 6
        adx = dx.ewm(span = m, min_periods = 1, ignore_na = True, adjust = False).mean()

        adxr = (adx + adx.shift(m)) / 2

        result = ((pos_di.shift(1) < neg_di.shift(1)) & (pos_di >= neg_di)) * 1\
                + ((pos_di.shift(1) > neg_di.shift(1)) & (pos_di <= neg_di)) * -1



    elif factor == "MTM":
        """
        动力指标
        mtm: 当日收盘价与n日前的收盘价的差
        mtma: mtm的移动平均

        To decide:
        计算mtm的步长n，目前设置为6
        计算mtma的周期长度m，目前设置为6
        """
        n = 6
        m = 6

        mtm = cls - cls.shift(n)
        mtma = mtm.rolling(window = m, min_periods = m-1).mean()

        result = ((mtm.shift(1) < mtma.shift(1)) & (mtm >= mtma)) * 1\
                +((mtm.shift(1) > mtma.shift(1)) & (mtm <= mtma)) * -1



    elif factor == "SAR":
        """
        抛物线指标，停损指标
        拟使用talib进行指标的计算
        """
        sar = cls.copy() * 0
        for each in sar.columns:
            try:
                res = talib.SAR(high[each], low[each], acceleration = 0.02, maximum = 0.2)
            except:
                res = np.nan * len(cls.index)
            sar[each] = res

        result = ((cls.shift(1) < sar.shift(1)) & (cls >= sar)) * 1 \
                +((cls.shift(1) > sar.shift(1)) & (cls <= sar)) * -1



    elif factor == "KDJ":
        """
        随机指标
        rsv：（第n日的收盘价与n日内的最低价的差 除以 n日内最高价与n日内最低价的差）乘以 100
        ln: n日内的最低价
        hn: n日内的最高价
        k值：2/3前一日k值 + 1/3当日RSV
        d值：2/3前一日d值 + 1/3当日k值
        j值：3*当日k值 - 2*当日d值

        To decide:
        计算RSI的区间长度n
        """
        n = 9
        ln, hn = low, high
        for i in range(1,n+1):
            ln = np.minimum(ln,low.shift(i))
            hn = np.maximum(hn,high.shift(i))

        rsv = (cls - ln) / (hn - ln) * 100

        k_value = cls.copy() * 0
        k_value.iloc[0] = 50
        k_value = k_value.fillna(50)

        d_value = cls.copy() * 0
        d_value.iloc[0] = 50
        d_value = d_value.fillna(50)

        for i in range(1,len(k_value)):
            k_value.iloc[i] = k_value.iloc[i-1] * 2/3 + rsv.iloc[i].fillna(50) / 3
            d_value.iloc[i] = d_value.iloc[i-1] * 2/3 + k_value.iloc[i] / 3

        j_value = 3 * k_value - 2 * d_value

        result = ((k_value <= 30) & (k_value >= 10) & (k_value.shift(1) < d_value.shift(1)) & (k_value >= d_value)) * 1\
                + ((k_value <= 90) & (k_value >= 70) & (k_value.shift(1) > d_value.shift(1)) & (k_value <= d_value)) * -1



    elif factor == "RSI":
        """
        相对强弱指标

        To decide:
        快速RSI的周期长度n，目前设置为12
        慢速RSI的周期长度m，目前设置为24
        """
        n = 12
        m = 24

        vol = (cls / opn - 1) * 100
        pos_vol = vol.copy() * 0
        neg_vol = vol.copy() * 0
        pos_vol[vol > 0] = vol[vol > 0]
        neg_vol[vol < 0] = vol[vol < 0]

        quick_rsi = pos_vol.rolling(window = n, min_periods = n - 2).mean() / \
                   (pos_vol.rolling(window = n, min_periods = n - 2).mean() + np.abs(neg_vol.rolling(window = n, min_periods = n - 2).mean()))

        slow_rsi = pos_vol.rolling(window = m, min_periods = m - n).mean() / \
                   (pos_vol.rolling(window = m, min_periods = m - n).mean() + np.abs(neg_vol.rolling(window = m, min_periods = m - n).mean()))

        result = ((quick_rsi <= 20) & (quick_rsi.shift(1) < slow_rsi.shift(1)) & (quick_rsi >= slow_rsi)) * 1\
                +((quick_rsi >= 80) & (quick_rsi.shift(1) > slow_rsi.shift(1)) & (quick_rsi <= slow_rsi)) * -1



    elif factor == "ROC":
        """
        变动率指标
        以今天的收盘价比较其n天前的收盘价的差除以n天前的收盘价

        To decide:
        周期n的选择，目前周期选择为5，市场上流行的通常是5或者10
        """

        n = 5
        roc = cls / cls.shift(n) - 1
        rocma = roc.rolling(window = n, min_periods = n-2).mean()

        result = (((trend != 0) & (roc.shift(1) < 0) & (roc >= 0)) | ((trend == 0) & (roc.shift(1) < rocma.shift(1)) & (roc >= rocma))) * 1\
                +(((trend != 0) & (roc.shift(1) > 0) & (roc <= 0)) | ((trend == 0) & (roc.shift(1) > rocma.shift(1)) & (roc <= rocma))) * -1



    elif factor == "B3612":
        """
        B36: 收盘价的3日移动平均线与6日移动平均线的乖离值
        B612: 收盘价的6日移动平均线与12日的移动平均线的乖离值
        """
        pass


    elif factor == "BIAS":
        """
        乖离率，计算收盘价与某条均线之间的差距百分比
        用6日，12日，24日乖离率进行判断
        """
        bias_short = cls / cls.rolling(window = 6, min_periods = 3).mean() - 1
        bias_middle = cls / cls.rolling(window = 12, min_periods = 6).mean() - 1
        bias_long = cls / (cls.rolling(window = 24, min_periods = 12).mean() ) - 1

        result = ((bias_short <= -0.04) | (bias_middle <= -0.05) | (bias_long <= -0.08)) * 1\
                +((bias_short >= 0.045) | (bias_middle >= 0.06) | (bias_long >= 0.09)) * -1



    elif factor == "CCI":
        """
        顺势指标
        tp: （最高价 + 最低价 + 收盘价）/ 3
        ma: n日tp价格的移动平均
        md: n日(MA - TP)的绝对值累积和的平均值
        cci: (tp - ma) / md / 0.015

        To decide:
        循环周期中n的值，系统默认为14
        """
        n =14

        tp = (high + low + cls) / 3
        ma = tp.rolling(window = n, min_periods = n-4).mean()
        md = (np.abs(ma - tp)).rolling(window = n, min_periods = n-4).mean()
        cci = (tp - ma) / md / 0.015

        result = (((cci.shift(1) < 100) & (cci >= 100)) | ((cci.shift(1) < -100) & (cci >= -100))) * 1\
                +(((cci.shift(1) > 100) & (cci <= 100)) | ((cci.shift(1) > -100) & (cci <= -100))) * -1



    elif factor == "OSC":
        """
        变动速率线

        To decide:
        计算均线的周期，一般为10日
        """
        n = 10

        osc = cls - cls.rolling(window = n, min_periods = n-2).mean()
        oscma = osc.rolling(window = n, min_periods = n -2).mean()

        result = ((osc.shift(1) < oscma.shift(1)) & (osc >= oscma)) * 1\
                +((osc.shift(1) > oscma.shift(1)) & (osc <= oscma)) * -1


    elif factor == "W&R":
        """
        威廉指标

        To decide:
        计算周期n，一般参数设置为10
        """
        n = 10

        wr = 100 * (high.rolling(window = n, min_periods = n-2).max() - cls) / (high.rolling(window = n, min_periods = n-2).max() - low.rolling(window = n, min_periods = n-2).min())
        result = ((wr.shift(1) < 80) & (wr >= 80)) * 1 + ((wr.shift(1) > 20) & (wr <= 20)) * -1



    elif factor == "SLOWKD":
        """
        慢速随机指标

        To decide:
        计算RSV的周期n，目前设置为9
        计算MARSV的周期m，目前设置为3
        """

        n = 9
        ln, hn = low, high
        for i in range(1,n+1):
            ln = np.minimum(ln,low.shift(i))
            hn = np.maximum(hn,high.shift(i))

        rsv = (cls - ln) / (hn - ln) * 100

        m = 3
        marsv = rsv.rolling(window = m, min_periods = m-2).mean()
        k_value = marsv.rolling(window = m, min_periods = m-2).mean()
        d_value = k_value.rolling(window = m, min_periods = m-2).mean()

        result = ((k_value <= 30) & (k_value >= 10) & (k_value.shift(1) < d_value.shift(1)) & (k_value > d_value)) * 1\
                +((k_value <= 90) & (k_value >= 70) & (k_value.shift(1) > d_value.shift(1)) & (k_value < d_value)) * -1


    elif factor == "MASS":
        """
        梅斯线
        dif: 最高价与最低价的差，名为交易区间
        ahl: dif的n天指数平均数，定义中为9
        bhl: ahl的n天指数平均数，定义中为9
        mass: ahl/bhl的m日的和，定义中为25
        ma: n天的股价平均线，定义中为9

        To decide:
        计算指数平均的周期n，定义中为9
        计算股价平均的周期n，定义中为9
        计算比值求和的周期m，定义中为25
        """
        n = 9
        m = 25
        dif = high - low
        ahl = dif.ewm(min_periods = 1, span = n, ignore_na = True , adjust = False).mean()
        bhl = ahl.ewm(min_periods = 1, span = n, ignore_na = True , adjust = False).mean()
        mass = (ahl/bhl).rolling(window = m, min_periods = m-n).sum()
        ma = cls.rolling(window = n, min_periods = n - 2).mean()

        result = ((mass.shift(2) > 27) & (mass.shift(1) <= 27) & (mass <=26.5) & (ma.shift(2) >= ma.shift(1)) & (ma.shift(1) >= ma)) * 1\
                +((mass.shift(2) < 27) & (mass.shift(1) >= 27) & (mass <=26.5) & (ma.shift(2) <= ma.shift(1)) & (ma.shift(1) <= ma)) * -1



    elif factor == "%B":
        """
        布林极限

        To decide：
        计算布林线上下轨的周期n，目前设置为20
        """
        n = 20
        up = cls.rolling(window = n, min_periods = n-5).mean() + 2 * cls.rolling(window = n, min_periods = n-5).std()
        down = cls.rolling(window = n, min_periods = n-5).mean() - 2 * cls.rolling(window = n, min_periods = n-5).std()
        bb = 100 * (cls - down) / (up - down)
        result = (bb < 0) * 1 + (bb > 100) * -1


    elif factor == "BBIBOLL":
        """
        多空布林线

        To decide:
        各个均线的window的值n,m,l,h，定义中为3，6，12，24
        计算bbiboll上下轨的系数k，目前定义为6
        计算bbiboll上下轨的窗口期s，目前定义为11
        计算bbibill的std的周期，目前定义为c = 200
        """
        n = 3
        m = 6
        l = 12
        h = 24
        ma3 = cls.rolling(window = n, min_periods = 1).mean()
        ma6 = cls.rolling(window = m, min_periods = m-1).mean()
        ma12 = cls.rolling(window = l, min_periods = l-5).mean()
        ma24 = cls.rolling(window = h, min_periods = h-5).mean()
        bbiboll = (ma3 + ma6 + ma12 + ma24)/4
        k = 6
        s = 11
        up = bbiboll + k * bbiboll.rolling(window = s, min_periods = s - 5).std()
        down = bbiboll - k * bbiboll.rolling(window = s, min_periods = s - 5).std()

        q3 = cls.rolling(window = 400, min_periods = 200).quantile(0.75)
        q1 = cls.rolling(window = 400, min_periods = 200).quantile(0.25)

        c = 200
        bbiboll_std = bbiboll.rolling(window = c, min_periods = c//2).std()
        std_q3 = bbiboll_std.rolling(window = 400, min_periods = 200).quantile(0.75)
        std_q1 = bbiboll_std.rolling(window = 400, min_periods = 200).quantile(0.25)

        result = ((cls < q1) & (cls.shift(1) < bbiboll.shift(1)) & (cls >= bbiboll) & (bbiboll_std < std_q1)) * 1\
                +((cls > q3) & (cls.shift(1) > bbiboll.shift(1)) & (cls <= bbiboll) & (bbiboll_std > std_q3)) * -1



    elif factor == "KELT":
        """
        未找到相应的定义和计算公式
        需要定一个初始价格，即建仓价格，定义为m个交易日的价格

        To decide:
        初始价格的周期，目前设置为m = 30
        """
        m = 30
        tr = np.maximum(np.maximum(np.abs(high - low), np.abs(high - cls.shift(1))), np.abs(low - cls.shift(1)))
        atr = tr.rolling(window = m, min_periods = m - 5).mean()
        result = (cls >= cls.shift(m) + 0.5 * atr) * 1 + (cls <= cls.shift(m) - 2 * atr) * -1


    elif factor == "ENV":
        """
        轨道线的简称

        To decide:
        计算价格线时，所选定的周期m，一般周期为14天，m = 14
        计算上下轨线时，在价格线基础上，向上下浮动的单位n，一般为，n = 0.06
        """
        m = 14
        n = 0.06
        up = (1 + n) * cls.rolling(window = m, min_periods = m-3).mean()
        down = (1 - n) * cls.rolling(window = m, min_periods = m-3).mean()

        result = ((cls.shift(1) > down.shift(1)) & (cls <= down)) * 1 + ((cls.shift(1) < up.shift(1)) & (cls >= up)) * -1


    elif factor == "CDP":
        """
        逆市操作指标，根据前一天的价格信息，在下一天同时卖出和买进股票，在一天之内对盯市进行操作，不符合目前的体系
        cdp: (前一日的最高价 + 最低价 + 2倍收盘价) / 4
        ah: cdp + (high - low)
        nh: cdp * 2 - low
        al: cdp - (high - low)
        nl: cdp * 2 - high
        """
        cdp = (high + low + 2 * cls) / 4
        ah = cdp + high - low
        nh = cdp * 2 - low
        al = cdp - (high - low)
        nl = cdp * 2 - high
        pass


    elif factor == "MIKE":
        """
        麦克支撑压力指标
        分为初级压力线和支撑线之间的轨道，中级压力线和支撑线之间的轨道和强力压力线和支撑线之间的轨道
        目前暂时用初级压力线之间的轨道来进行信号的判断
        typ: （当日最高 + 当日最低 + 当日收盘价）/ 3
        hn : n日的最高价的最高值
        ln : n日内最低价的最低值
        wekr: 初级压力线 = typ + typ - ln
        midr: 中级压力线 = typ + hn - ln
        stor: 强力压力线 = 2 * hn - ln
        weks: 初级支撑线 = typ - (hn - typ)
        mids: 中级支撑线 = typ - (hn - ln)
        stos: 强力支撑线 = 2 * ln - hn

        To decide:
        计算最高价和最低价的窗口期n,目前为9
        计算买入和卖出信号的轨道，目前是初级压力线和中级支撑线之间的轨道
        """
        typ = (high + low + cls) / 3

        n = 9
        ln, hn = low, high
        for i in range(1,n+1):
            ln = np.minimum(ln,low.shift(i))
            hn = np.maximum(hn,high.shift(i))

        wekr = typ + (typ - ln)
        midr = typ + (hn - ln)
        stor = 2 * hn - ln

        weks = typ - (hn - typ)
        mids = typ - (hn - ln)
        stos = 2 * ln - hn

        result = ((long_trend == 1) & (cls.shift(1) < wekr.shift(1)) & (cls >= wekr)) * 1\
               +((long_trend == -1) & (cls.shift(1) > weks.shift(1)) & (cls <= weks)) * -1\
                +((long_trend == 0) & (cls.shift(1) > weks.shift(1)) & (cls <= weks)) * 1\
                +((long_trend == 0) & (cls.shift(1) < wekr.shift(1)) & (cls >= wekr)) * -1


    elif factor == "CHAIKIN":
        """
        暂时未找到相应公式
        """
        pass


    elif factor == "OBV":
        """
        量能累积线，每日的成交量的累积，若上涨则为正的成交量，若下降则为负的成交量

        To decide:
        研报中创出新高的窗口期n的值，目前定义为5
        """
        obv = volume.copy() * 0
        obv[cls >= opn] = volume[cls >= opn]
        obv[cls < open] = -1 * volume[cls < opn]
        obv = obv.cumsum()

        n = 5

        high_obv = obv.rolling(window = n, min_periods = n -2).max()
        low_obv = obv.rolling(window = n, min_periods = n -2).min()
        high_cls = cls.rolling(window = n, min_periods = n -2).max()
        low_cls = cls.rolling(window = n, min_periods = n -2).min()

        res_buy = (((cls == low_cls) & (obv != low_obv)) | ((cls != low_cls) & (obv == low_obv)) | ((obv.shift(1) < 0) & (obv >= 0))) * 1
        res_sell = (((cls == high_cls) & (obv != high_obv)) | ((cls != high_cls) & (obv == high_obv)) | ((obv.shift(1) > 0) & (obv <= 0))) * -1

        result = res_buy + res_sell


    elif factor == "EMV":
        """
        简易波动指标
        a = (今日最高 + 今日最低) / 2
        b = (前日最高 + 前日最低) / 2
        c = (今日最高 - 今日最低)
        em = (a - b) * c / 今日成交额
        emv = n日内em的累和
        maemv = emv的m日的简单移动平均

        To decide:
        参数m的值，目前为 m=9; 参数n的值，目前为 n=14;
        emv趋向于0的定义
        """
        m = 9
        n = 14
        a = (high + low) / 2
        b = (high.shift(1) + low.shift(1)) / 2
        c = high - low
        em = (a - b) * c / amount
        emv = em.rolling(window = n, min_periods = n - 5).sum()
        maemv = emv.rolling(window = m, min_periods = m -2).mean()

        result = ((emv > 0) & (emv <= 0.05)) * 1 + ((emv < 0) & (emv >= -0.05)) * -1



    elif factor == "TAPI":
        """
        现值率
        研报中未给出计算公式和具体的用法，只说了此因子不单独使用，要与大势和K线图等一起使用，故不进行计算
        """
        pass

    return result
