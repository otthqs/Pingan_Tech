# -*- coding:utf-8 -*-
from tool import *

fst = ["003","004","006","013","014","015","016","024","026","027","028","031","032","036","040","042","043","044","047","050","055","066","094"]

def calculate_volume_price_factor(name):
    if name == "alpha003":
        result = -1 * correlation(rank(opn), rank(volume), 10)

    elif name == "alpha004":
        result = -1 * ts_rank(rank(low), 9)

    elif name == "alpha006":
        result = -1 * correlation(opn, volume, 10)

    elif name == "alpha013":
        result = -1 * rank(covariance(rank(cls), rank(volume),5))

    elif name == "alpha014":
        result = (-1 * rank(delta(returns,3))) * correlation(opn,volume,10)

    elif name == "alpha015":
        result = -1 * sum_(rank(correlation(rank(high), rank(volume), 3)), 3)

    elif name == "alpha016":
        result = -1 * rank(covariance(rank(high), rank(volume), 5))

    elif name == "alpha024":
        result = con((((delta((sum_(cls,100)/100),100)/delay(cls,100))<0.05)|((delta((sum_(cls,100)/100),100)/delay(cls,100))==0.05)), (-1*(cls-ts_min (cls,100))), (-1*delta(cls,3)))

    elif name == "alpha026":
        result = -1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)

    elif name == "alpha027":
        result = con((0.5 < rank((sum_(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))), -1, 1)

    elif name == "alpha028":
        result = scale(((correlation(get_adv(volume, 20), low, 5) + ((high + low) / 2)) - cls))

    elif name == "alpha031":
        result = (rank(rank(rank(decay_linear((-1 * rank(rank(delta(cls, 10)))),10)))) + rank((-1 * delta(cls, 3)))) + sign(scale(correlation(adv20,low, 12)))

    elif name == "alpha032":
        result = scale(((sum_(cls, 7)/7) - cls)) + (20 * scale(correlation(vwap, delay(cls, 5),230)))

    elif name == "alpha036":
        result = ((((2.21 * rank(correlation((cls - opn), delay(volume, 1), 15))) + (0.7 * rank((opn - cls)))) + (0.73 * rank(ts_rank(delay((-1 * returns),6),5)))) + rank(abs(correlation(vwap, get_adv(volume, 20), 6)))) + (0.6 * rank((((sum_(cls, 200)/200) - opn) * (cls - opn))))

    elif name == "alpha040":
        result = (-1 * rank(stddev(high, 10))) * correlation(high, volume, 10)

    elif name == "alpha042":
        result = rank((vwap - cls)) / rank((vwap + cls))

    elif name == "alpha043":
        result = ts_rank((volume / get_adv(volume, 20)), 20) * ts_rank((-1 * delta(cls, 7)), 8)

    elif name == "alpha044":
        result = -1 * correlation(high, rank(volume), 5)

    elif name == "alpha047":
        result = (((rank((1 / cls)) * volume)/ get_adv(volume, 20)) * ((high * rank((high - cls))) / (sum_(high, 5) / 5))) - rank((vwap - delay(vwap, 5)))

    elif name == "alpha050":
        result = -1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)

    elif name == "alpha055":
        result = -1 * correlation(rank(((cls - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6)

    elif name == "alpha066":
        result = -1 * (rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + ts_rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (opn - ((high + low) / 2))),11.4157),6.72611))


    elif name == "alpha094":
        adv60 = get_adv(volume, 60)
        result = -1 * (rank((vwap - ts_min(vwap, 11.5783))) ** ts_rank(
                correlation(ts_rank(vwap, 19.6462), ts_rank(adv60, 4.02992), 18.0926), 2.70756))

    return result
