import pandas as pd
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
import numpy as np
import logging


def compensation(hist_actual, hist_pred, pred, period=7, steps_day=7):
    """
    :param hist_actual: 近期销售，任何一维数据
    :param hist_pred: 近期预测值，任何一维数据
    :param pred: 已发布的未来预测值，任何一维数据
    :param period: HoltWinter方法的周期，此处取周季节性
    :param steps_day: 预测天数
    :return: 加载补偿量的最终预测值
    """
    hist_actual = pd.Series(hist_actual[:29])
    hist_pred = pd.Series(hist_pred[:29])
    pred = pd.Series(pred[:steps_day])
    deviation = hist_pred - hist_actual
    weights = []
    for i in range(1, len(deviation) + 1):
        weights.append(i / len(deviation))
    weights = np.array(weights) / sum(weights)
    if len(deviation) <= 3:
        logging('历史数据过少，不进行偏差量补偿')
        final_pred = pred
    elif 3 < len(deviation) <= 14:
        # --------------------------------------------SES-------------------------------------------------------------
        # 拟合历史数据
        fit_SES = SimpleExpSmoothing(deviation, initialization_method='known', initial_level=np.average(deviation, weights=weights))\
            .fit(smoothing_level=1 / 10, optimized=False, use_boxcox=None, remove_bias=False)
        fit_SES_train = SimpleExpSmoothing(deviation, initialization_method='known', initial_level=np.average(deviation, weights=weights))\
            .fit(smoothing_level=1 / 5, optimized=False, use_boxcox=None, remove_bias=False)
        # ----------------------------------------Holt------------------------------------------------------------
        # fit models
        Holt_add_dam = Holt(deviation, exponential=False, damped_trend=True, initialization_method='known',
                            initial_level=np.average(deviation, weights=weights),
        initial_trend=np.array((sum(deviation[int(np.ceil(len(deviation) / 2)):]) - sum(deviation[:int(np.floor(len(deviation) / 2))])) / (np.floor(len(deviation) / 2)) ** 2))\
            .fit(smoothing_level=1 / 10, smoothing_trend=1 / 10, damping_trend=0.90, optimized=False, use_boxcox=None, remove_bias=False)
        Holt_add_dam_train = Holt(deviation, exponential=False, damped=True, initialization_method='known',
                            initial_level=np.average(deviation,
                                                     weights=weights),
                            initial_trend=np.array((sum(deviation[
                                                        int(
                                                            np.ceil(
                                                                len(
                                                                    deviation) / 2)):]) - sum(
                                deviation[:int(np.floor(
                                    len(deviation) / 2))])) / (
                                                       np.floor(
                                                           len(
                                                               deviation) / 2)) ** 2)).fit(smoothing_level=1 / 5,
                                                                                 smoothing_trend=1 / 5,
                                                                                 damping_trend=0.85, optimized=False, use_boxcox=None,  remove_bias = False)

        # 总补偿量为正，则按预测值的正比重新分配每步补偿量，总补偿量为负，则按预测值的反比重新分配每步补偿量；使补偿后的预测值更加稳定，并在多步补偿的情况下与实际值更接近。
        pred = np.array(pred)  # pd.Series是按对应索引进行运算，np.array是按对应顺序进行运算
        ratio_posi = pred / sum(pred)
        ratio_posi[ratio_posi < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_posi = ratio_posi / sum(ratio_posi)
        ratio_neg = 1 / 2 * (1 - ratio_posi) / sum(1 - ratio_posi) + 1 / 2 * (1 / ratio_posi) / sum(1 / ratio_posi)
        ratio_neg[ratio_neg < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_neg = ratio_neg / sum(ratio_neg)
        total_compensation = sum(4 / 5 * np.average(
            np.column_stack([np.array(Holt_add_dam.forecast(steps_day)).reshape(-1, 1), np.array(
                Holt_add_dam_train.forecast(steps_day)).reshape(-1, 1)]), axis=1) + 1 / 5 * np.average(
            np.column_stack([np.array(fit_SES.forecast(steps_day)).reshape(-1, 1), np.array(
                fit_SES_train.forecast(steps_day)).reshape(-1, 1)]), axis=1))
        if total_compensation >= 0:
            every_compensation = total_compensation * ratio_posi
        else:
            every_compensation = total_compensation * ratio_neg
        final_pred = pd.Series(pred - every_compensation)

    else:
        # -------------------------------------------------Holt--------------------------------------------------
        # fit models
        Holt_add_dam = Holt(deviation, exponential=False, damped_trend=True, initialization_method='known',
                            initial_level=np.average(deviation,
                                                     weights=weights),
                            initial_trend=np.array((sum(deviation[
                                                        int(
                                                            np.ceil(
                                                                len(
                                                                    deviation) / 2)):]) - sum(
                                deviation[:int(np.floor(
                                    len(deviation) / 2))])) / (
                                                       np.floor(
                                                           len(
                                                               deviation) / 2)) ** 2)).fit(smoothing_level=1 / 10,
            smoothing_trend=1 / 10, damping_trend=0.90, optimized=False, use_boxcox=None, remove_bias=False)
        Holt_add_dam_train = Holt(deviation, exponential=False, damped=True, initialization_method='known',
                            initial_level=np.average(deviation,
                                                     weights=weights),
                            initial_trend=np.array((sum(deviation[
                                                        int(
                                                            np.ceil(
                                                                len(
                                                                    deviation) / 2)):]) - sum(
                                deviation[:int(np.floor(
                                    len(deviation) / 2))])) / (
                                                       np.floor(
                                                           len(
                                                               deviation) / 2)) ** 2)).fit(smoothing_level=1 / 5,
                                                                                 smoothing_trend=1 / 5,
                                                                                 damping_trend=0.85, optimized=False, use_boxcox=None,  remove_bias = False)
        # ---------------------------------------------------HoltWinters-----------------------------------------------
        # fit models
        HW_add_add_dam = ExponentialSmoothing(deviation, trend='add', damped_trend=True, seasonal='add',
                                              seasonal_periods=period, initialization_method='known',
                                              initial_level=np.average(deviation, weights=weights),
                                              initial_trend=np.array((sum(
                                                  deviation[int(np.ceil(len(deviation) / 2)):]) - sum(
                                                  deviation[:int(np.floor(len(deviation) / 2))])) / (
                                                                         np.floor(len(deviation) / 2)) ** 2),
                                              initial_seasonal=np.array(
                                                  deviation[:len(deviation) // period * period]).reshape(-1,
                                                                                                         period).mean(
                                                  axis=0) - np.average(deviation, weights=weights), use_boxcox=None, missing='raise'). \
            fit(smoothing_level=1 / 10, smoothing_trend=1 / 10, smoothing_seasonal=1 / 2, damping_trend=0.90, optimized=False, remove_bias=False)
        HW_add_add_dam_train = ExponentialSmoothing(deviation, trend='add', damped_trend=True, seasonal='add',
                                                    seasonal_periods=period, initialization_method='known',
                                                    initial_level=np.average(deviation, weights=weights),
                                                    initial_trend=np.array((sum(
                                                        deviation[int(np.ceil(len(deviation) / 2)):]) - sum(
                                                        deviation[:int(np.floor(len(deviation) / 2))])) / (
                                                                               np.floor(len(deviation) / 2)) ** 2),
                                                    initial_seasonal=np.array(
                                                        deviation[:len(deviation) // period * period]).reshape(-1,
                                                                                                               period).mean(
                                                        axis=0) - np.average(deviation, weights=weights), use_boxcox=None, missing='raise'). \
            fit(smoothing_level=1 / 5, smoothing_trend=1 / 10, smoothing_seasonal=1 / 10,
                damping_trend=0.85, optimized=False, remove_bias=False)

        # 总补偿量为正，则按预测值的正比重新分配每步补偿量，总补偿量为负，则按预测值的反比重新分配每步补偿量；使补偿后的预测值更加稳定，并在多步补偿的情况下与实际值更接近。
        pred = np.array(pred)  # pd.Series是按对应索引进行运算，np.array是按对应顺序进行运算
        ratio_posi = pred / sum(pred)
        ratio_posi[ratio_posi < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_posi = ratio_posi / sum(ratio_posi)
        ratio_neg = 1 / 2 * (1 - ratio_posi) / sum(1 - ratio_posi) + 1 / 2 * (1 / ratio_posi) / sum(1 / ratio_posi)
        ratio_neg[ratio_neg < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_neg = ratio_neg / sum(ratio_neg)
        total_compensation = sum(1 / 2 * np.average(
            np.column_stack([np.array(Holt_add_dam.forecast(steps_day)).reshape(-1, 1), np.array(
                Holt_add_dam_train.forecast(steps_day)).reshape(-1, 1)]), axis=1) + 1 / 2 * np.average(
            np.column_stack([np.array(HW_add_add_dam.forecast(steps_day)).reshape(-1, 1), np.array(
                HW_add_add_dam_train.forecast(steps_day)).reshape(-1, 1)]), axis=1))
        if total_compensation >= 0:
            every_compensation = total_compensation * ratio_posi
        else:
            every_compensation = total_compensation * ratio_neg
        final_pred = pred - pd.Series(every_compensation)

    final_pred[final_pred<0] = 0
    return final_pred


def compensation_first(hist_actual, hist_pred, pred, period=7):
    """
    :param hist_actual: 近期理论销售历史值，任何一维数据，转化为pd.Series，为了作图观察
    :param hist_pred: 近期倒数第二次预测值，加载老r、效应系数、新r，但不加载补偿量，任何一维数据，转化为pd.Series，为了作图观察
    :param pred: 未来7天的倒数第二次预测值，加载老r、效应系数、新r，但不加载补偿量，任何一维数据，转化为pd.Series，为了作图观察
    :param period: HoltWinter方法的周期
    :return: 加载补偿量的最终预测值，np.array，使其无索引
    """
    if len(pred) < 2:
        return pred - pred
        # raise Exception('pred长度必须≥2')
    steps_day = len(pred)  # 补偿天数
    hist_actual = pd.Series(hist_actual[-29:])
    hist_pred = pd.Series(hist_pred[-len(hist_actual):])
    pred = pd.Series(pred)
    deviation = hist_pred - hist_actual
    weights = []
    for i in range(1, len(deviation) + 1):
        weights.append(i / len(deviation))
    weights = np.array(weights) / sum(weights)
    if len(deviation) <= 3:
        logging.warning('historical data are extremely few')
        # LogFile('历史数据过少，不进行偏差量补偿')
        every_compensation = [0] * steps_day
    elif 4 <= len(deviation) <= 14:
        # SES
        # 拟合历史数据
        fit_SES = SimpleExpSmoothing(deviation,
                                     initialization_method='known',
                                     initial_level=np.average(deviation, weights=weights)).fit(smoothing_level=1 / 10,
                                                                                               optimized=False,
                                                                                               start_params=1 / 10,
                                                                                               use_brute=False,
                                                                                               use_boxcox=False,
                                                                                               remove_bias=False)
        fit_SES_train = SimpleExpSmoothing(deviation, initialization_method='known', initial_level=np.average(deviation,
                                                                                                              weights=weights)).fit(
            smoothing_level=1 / 5, optimized=False, start_params=1 / 5, use_brute=False, use_boxcox=False,
            remove_bias=False)
        # Holt
        # fit models
        Holt_add_dam = Holt(deviation, exponential=False, damped_trend=True, initialization_method='known',
                            initial_level=np.average(deviation, weights=weights),
                            initial_trend=np.array((sum(deviation[int(np.ceil(len(deviation) / 2)):]) -
                                                    sum(deviation[:int(np.floor(len(deviation) / 2))])) /
                                                   (np.floor(len(deviation) / 2)) ** 2)).fit(smoothing_level=1 / 10,
                                                                                             smoothing_trend=1 / 10,
                                                                                             damping_trend=0.90,
                                                                                             optimized=False,
                                                                                             start_params=[1 / 10,
                                                                                                           1 / 10],
                                                                                             use_brute=False,
                                                                                             use_boxcox=False,
                                                                                             remove_bias=False)
        Holt_add_dam_train = Holt(deviation, exponential=False, damped_trend=True, initialization_method='known',
                                  initial_level=np.average(deviation,
                                                           weights=weights),
                                  initial_trend=np.array((sum(deviation[
                                                              int(
                                                                  np.ceil(
                                                                      len(
                                                                          deviation) / 2)):]) - sum(
                                      deviation[:int(np.floor(
                                          len(deviation) / 2))])) / (
                                                             np.floor(
                                                                 len(
                                                                     deviation) / 2)) ** 2)).fit(smoothing_level=1 / 5,
                                                                                                 smoothing_trend=1 / 5,
                                                                                                 damping_trend=0.85,
                                                                                                 optimized=False,
                                                                                                 start_params=[1 / 5,
                                                                                                               1 / 5],
                                                                                                 use_brute=False,
                                                                                                 use_boxcox=False,
                                                                                                 remove_bias=False)

        # 总补偿量为正，则按预测值的正比重新分配每步补偿量，总补偿量为负，则按预测值的反比重新分配每步补偿量；使补偿后的预测值更加稳定，并在多步补偿的情况下与实际值更接近。
        pred = np.array(pred)  # pd.Series是按对应索引进行运算，np.array是按对应顺序进行运算
        ratio_posi = pred / sum(pred)
        ratio_posi[ratio_posi < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_posi = ratio_posi / sum(ratio_posi)
        ratio_neg = 1 / 2 * (1 - ratio_posi) / sum(1 - ratio_posi) + 1 / 2 * (1 / ratio_posi) / sum(1 / ratio_posi)
        ratio_neg[ratio_neg < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_neg = ratio_neg / sum(ratio_neg)
        total_compensation = sum(4 / 5 * np.average(
            np.column_stack([np.array(Holt_add_dam.forecast(steps_day)).reshape(-1, 1), np.array(
                Holt_add_dam_train.forecast(steps_day)).reshape(-1, 1)]), axis=1) + 1 / 5 * np.average(
            np.column_stack([np.array(fit_SES.forecast(steps_day)).reshape(-1, 1), np.array(
                fit_SES_train.forecast(steps_day)).reshape(-1, 1)]), axis=1))
        if total_compensation >= 0:
            every_compensation = total_compensation * ratio_posi
        else:
            every_compensation = total_compensation * ratio_neg

    else:
        # Holt
        # fit models
        Holt_add_dam = Holt(deviation, exponential=False, damped_trend=True, initialization_method='known',
                            initial_level=np.average(deviation,
                                                     weights=weights),
                            initial_trend=np.array((sum(deviation[
                                                        int(
                                                            np.ceil(
                                                                len(
                                                                    deviation) / 2)):]) - sum(
                                deviation[:int(np.floor(
                                    len(deviation) / 2))])) / (
                                                       np.floor(
                                                           len(
                                                               deviation) / 2)) ** 2)).fit(smoothing_level=1 / 10,
                                                                                           smoothing_trend=1 / 10,
                                                                                           damping_trend=0.90,
                                                                                           optimized=False,
                                                                                           start_params=[1 / 10,
                                                                                                         1 / 10],
                                                                                           use_brute=False,
                                                                                           use_boxcox=False,
                                                                                           remove_bias=False)
        Holt_add_dam_train = Holt(deviation, exponential=False, damped_trend=True, initialization_method='known',
                                  initial_level=np.average(deviation,
                                                           weights=weights),
                                  initial_trend=np.array((sum(deviation[
                                                              int(
                                                                  np.ceil(
                                                                      len(
                                                                          deviation) / 2)):]) - sum(
                                      deviation[:int(np.floor(
                                          len(deviation) / 2))])) / (
                                                             np.floor(
                                                                 len(
                                                                     deviation) / 2)) ** 2)).fit(smoothing_level=1 / 5,
                                                                                                 smoothing_trend=1 / 5,
                                                                                                 damping_trend=0.85,
                                                                                                 optimized=False,
                                                                                                 start_params=[1 / 5,
                                                                                                               1 / 5],
                                                                                                 use_brute=False,
                                                                                                 use_boxcox=False,
                                                                                                 remove_bias=False)
        # HoltWinters
        # fit models
        HW_add_add_dam = ExponentialSmoothing(deviation, trend='add', damped_trend=True,
                                              seasonal='add', seasonal_periods=period, initialization_method='known',
                                              initial_level=np.average(deviation, weights=weights),
                                              initial_trend=np.array((sum(
                                                  deviation[int(np.ceil(len(deviation) / 2)):]) - sum(
                                                  deviation[:int(np.floor(len(deviation) / 2))])) / (
                                                                         np.floor(len(deviation) / 2)) ** 2),
                                              initial_seasonal=np.array(
                                                  deviation[:len(deviation) // period * period]).reshape(-1,
                                                                                                         period).mean(
                                                  axis=0) - np.average(deviation, weights=weights), use_boxcox=False,
                                              missing='raise'). \
            fit(smoothing_level=1 / 10, smoothing_trend=1 / 10, smoothing_seasonal=1 / 5,
                damping_trend=0.90, optimized=False, remove_bias=False, start_params=[1 / 10, 1 / 10, 1 / 5],
                use_brute=False)
        HW_add_add_dam_train = ExponentialSmoothing(deviation, trend='add', damped_trend=True, seasonal='add',
                                                    seasonal_periods=period, initialization_method='known',
                                                    initial_level=np.average(deviation, weights=weights),
                                                    initial_trend=np.array((sum(
                                                        deviation[int(np.ceil(len(deviation) / 2)):]) - sum(
                                                        deviation[:int(np.floor(len(deviation) / 2))])) / (
                                                                               np.floor(len(deviation) / 2)) ** 2),
                                                    initial_seasonal=np.array(
                                                        deviation[:len(deviation) // period * period]).reshape(-1,
                                                                                                               period).mean(
                                                        axis=0) - np.average(deviation, weights=weights),
                                                    use_boxcox=False, missing='raise'). \
            fit(smoothing_level=1 / 5, smoothing_trend=1 / 5, smoothing_seasonal=1 / 3,
                damping_trend=0.85, optimized=False, remove_bias=False, start_params=[1 / 5, 1 / 5, 1 / 3],
                use_brute=False)

        # 总补偿量为正，则按预测值的正比重新分配每步补偿量，总补偿量为负，则按预测值的反比重新分配每步补偿量；使补偿后的预测值更加稳定，并在多步补偿的情况下与实际值更接近。
        pred = np.array(pred)  # pd.Series是按对应索引进行运算，np.array是按对应顺序进行运算
        ratio_posi = pred / sum(pred)
        ratio_posi[ratio_posi < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_posi = ratio_posi / sum(ratio_posi)
        ratio_neg = 1 / 2 * (1 - ratio_posi) / sum(1 - ratio_posi) + 1 / 2 * (1 / ratio_posi) / sum(1 / ratio_posi)
        ratio_neg[ratio_neg < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_neg = ratio_neg / sum(ratio_neg)
        total_compensation = sum(1 / 2 * np.average(
            np.column_stack([np.array(Holt_add_dam.forecast(steps_day)).reshape(-1, 1), np.array(
                Holt_add_dam_train.forecast(steps_day)).reshape(-1, 1)]), axis=1) + 1 / 2 * np.average(
            np.column_stack([np.array(HW_add_add_dam.forecast(steps_day)).reshape(-1, 1), np.array(
                HW_add_add_dam_train.forecast(steps_day)).reshape(-1, 1)]), axis=1))
        if total_compensation >= 0:
            every_compensation = total_compensation * ratio_posi
        else:
            every_compensation = total_compensation * ratio_neg

    return every_compensation


def compensation_first(hist_actual, hist_pred, pred, period=7):
    """
    :param hist_actual: 近期理论销售历史值，任何一维数据，转化为pd.Series，为了作图观察
    :param hist_pred: 近期倒数第二次预测值，加载老r、效应系数、新r，但不加载补偿量，任何一维数据，转化为pd.Series，为了作图观察
    :param pred: 未来7天的倒数第二次预测值，加载老r、效应系数、新r，但不加载补偿量，任何一维数据，转化为pd.Series，为了作图观察
    :param period: HoltWinter方法的周期
    :param steps_day: 预测天数
    :return: 加载补偿量的最终预测值，np.array，使其无索引
    """
    if len(pred) < 2:

        return pred - pred
        # raise Exception('pred长度必须≥2')
    steps_day = len(pred)  # 补偿天数
    hist_actual = pd.Series(hist_actual[-22:])
    hist_pred = pd.Series(hist_pred[-len(hist_actual):])
    pred = pd.Series(pred)
    deviation = hist_pred - hist_actual
    weights = []
    for i in range(1, len(deviation) + 1):
        weights.append(i / len(deviation))
    weights = np.array(weights)**4 / sum(np.array(weights)**4)  # 以幂函数的方式使距当前越近的点对initial_level所占权重越大

    if len(deviation) <= 3:
        logging.warning('historical data are extremely few')
        # LogFile('历史数据过少，不进行偏差量补偿')
        every_compensation = [0] * steps_day

    elif 3 < len(deviation) <= 14:
        fit_SES = SimpleExpSmoothing(deviation, initialization_method='known', initial_level=np.average(deviation,
            weights=weights)).fit(smoothing_level=2 / 3, optimized=False, use_brute=False, use_boxcox=False, remove_bias=False)
        fit_SES_train = SimpleExpSmoothing(deviation, initialization_method='known', initial_level=np.average(deviation,
            weights=weights)).fit(smoothing_level=1 / 2, optimized=False, use_brute=False, use_boxcox=False, remove_bias=False)


        Holt_add_dam = Holt(deviation, exponential=False, damped_trend=True, initialization_method='known',
                            initial_level=np.average(deviation,
                                                     weights=weights),
                            initial_trend=np.array((sum(deviation[
                                                        int(
                                                            np.ceil(
                                                                len(
                                                                    deviation) / 2)):]) - sum(
                                deviation[:int(np.floor(
                                    len(deviation) / 2))])) / (
                                                       np.floor(
                                                           len(
                                                               deviation) / 2)) ** 2)).fit(smoothing_level=2 / 3,
            smoothing_trend=1 / 6, damping_trend=1.02, optimized=False, use_brute=False,
            use_boxcox=False, remove_bias=False)
        Holt_add_dam_train = Holt(deviation, exponential=False, damped_trend=True, initialization_method='known',
                            initial_level=np.average(deviation,
                                                     weights=weights),
                            initial_trend=np.array((sum(deviation[
                                                        int(
                                                            np.ceil(
                                                                len(
                                                                    deviation) / 2)):]) - sum(
                                deviation[:int(np.floor(
                                    len(deviation) / 2))])) / (
                                                       np.floor(
                                                           len(
                                                               deviation) / 2)) ** 2)).fit(smoothing_level=1/2,
                                                                                 smoothing_trend=1/8,
                                                                                 damping_trend=1.05, optimized=False,
        use_brute = False, use_boxcox = False, remove_bias = False)


        # 总补偿量为正，则按预测值的正比重新分配每步补偿量，总补偿量为负，则按预测值的反比重新分配每步补偿量；使补偿后的预测值更加稳定，并在多步补偿的情况下与实际值更接近。
        pred = np.array(pred)  # pd.Series是按对应索引进行运算，np.array是按对应顺序进行运算
        ratio_posi = pred / sum(pred)
        ratio_posi[ratio_posi < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_posi = ratio_posi / sum(ratio_posi)
        ratio_neg = 1 / 2 * (1 - ratio_posi) / sum(1 - ratio_posi) + 1 / 2 * (1 / ratio_posi) / sum(1 / ratio_posi)
        ratio_neg[ratio_neg < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_neg = ratio_neg / sum(ratio_neg)
        total_compensation = sum(4 / 5 * np.average(
            np.column_stack([np.array(Holt_add_dam.forecast(steps_day)).reshape(-1, 1), np.array(
                Holt_add_dam_train.forecast(steps_day)).reshape(-1, 1)]), axis=1) + 1 / 5 * np.average(
            np.column_stack([np.array(fit_SES.forecast(steps_day)).reshape(-1, 1), np.array(
                fit_SES_train.forecast(steps_day)).reshape(-1, 1)]), axis=1))
        if total_compensation >= 0:
            every_compensation = total_compensation * ratio_posi
        else:
            every_compensation = total_compensation * ratio_neg

    else:
        Holt_add_dam = Holt(deviation, exponential=False, damped_trend=True, initialization_method='known',
                            initial_level=np.average(deviation,
                                                     weights=weights),
                            initial_trend=np.array((sum(deviation[
                                                        int(
                                                            np.ceil(
                                                                len(
                                                                    deviation) / 2)):]) - sum(
                                deviation[:int(np.floor(
                                    len(deviation) / 2))])) / (
                                                       np.floor(
                                                           len(
                                                               deviation) / 2)) ** 2)).fit(smoothing_level=2/3,
            smoothing_trend=1/6, damping_trend=1.02, optimized=False, use_brute=False,
            use_boxcox=False, remove_bias=False)
        Holt_add_dam_train = Holt(deviation, exponential=False, damped_trend=True, initialization_method='known',
                            initial_level=np.average(deviation,
                                                     weights=weights),
                            initial_trend=np.array((sum(deviation[
                                                        int(
                                                            np.ceil(
                                                                len(
                                                                    deviation) / 2)):]) - sum(
                                deviation[:int(np.floor(
                                    len(deviation) / 2))])) / (
                                                       np.floor(
                                                           len(
                                                               deviation) / 2)) ** 2)).fit(smoothing_level=1/2,
                                                                                 smoothing_trend=1/8,
                                                                                 damping_trend=1.05, optimized=False,
        use_brute = False, use_boxcox = False, remove_bias = False)


        HW_add_add_dam = ExponentialSmoothing(deviation, trend='add', damped_trend=True,
                                              seasonal='add', seasonal_periods=period, initialization_method='known',
                                              initial_level=np.average(deviation, weights=weights),
                                              initial_trend=np.array((sum(
                                                  deviation[int(np.ceil(len(deviation) / 2)):]) - sum(
                                                  deviation[:int(np.floor(len(deviation) / 2))])) / (
                                                                         np.floor(len(deviation) / 2)) ** 2),
                                              initial_seasonal=np.array(
                                                  deviation[:len(deviation) // period * period]).reshape(-1,
                                                                                                         period).mean(
                                                  axis=0) - np.average(deviation, weights=weights), use_boxcox=False, missing='raise'). \
            fit(smoothing_level=2/3, smoothing_trend=1/6, smoothing_seasonal=2/3,
                damping_trend=1.02, optimized=False, remove_bias=False, use_brute=False)
        HW_add_add_dam_train = ExponentialSmoothing(deviation, trend='add', damped_trend=True, seasonal='add',
                                                    seasonal_periods=period, initialization_method='known',
                                                    initial_level=np.average(deviation, weights=weights),
                                                    initial_trend=np.array((sum(
                                                        deviation[int(np.ceil(len(deviation) / 2)):]) - sum(
                                                        deviation[:int(np.floor(len(deviation) / 2))])) / (
                                                                               np.floor(len(deviation) / 2)) ** 2),
                                                    initial_seasonal=np.array(
                                                        deviation[:len(deviation) // period * period]).reshape(-1,
                                                                                                               period).mean(
                                                        axis=0) - np.average(deviation, weights=weights), use_boxcox=False, missing='raise'). \
            fit(smoothing_level=1/2, smoothing_trend=1/8, smoothing_seasonal=1/2,
                damping_trend=1.05, optimized=False, remove_bias=False, use_brute=False)

        # 总补偿量为正，则按预测值的正比重新分配每步补偿量，总补偿量为负，则按预测值的反比重新分配每步补偿量；使补偿后的预测值更加稳定，并在多步补偿的情况下与实际值更接近。
        pred = np.array(pred)  # pd.Series是按对应索引进行运算，np.array是按对应顺序进行运算
        ratio_posi = pred / sum(pred)
        ratio_posi[ratio_posi < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_posi = ratio_posi / sum(ratio_posi)
        ratio_neg = 1 / 2 * (1 - ratio_posi) / sum(1 - ratio_posi) + 1 / 2 * (1 / ratio_posi) / sum(1 / ratio_posi)
        ratio_neg[ratio_neg < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_neg = ratio_neg / sum(ratio_neg)
        total_compensation = sum(1 / 2 * np.average(
            np.column_stack([np.array(Holt_add_dam.forecast(steps_day)).reshape(-1, 1), np.array(
                Holt_add_dam_train.forecast(steps_day)).reshape(-1, 1)]), axis=1) + 1 / 2 * np.average(
            np.column_stack([np.array(HW_add_add_dam.forecast(steps_day)).reshape(-1, 1), np.array(
                HW_add_add_dam_train.forecast(steps_day)).reshape(-1, 1)]), axis=1))
        if total_compensation >= 0:
            every_compensation = total_compensation * ratio_posi
        else:
            every_compensation = total_compensation * ratio_neg

    return every_compensation

def compensation_sec(hist_actual, hist_pred, pred, period=7):
    """
    :param hist_actual: 近期理论销售历史值，任何一维数据，转化为pd.Series，为了作图观察
    :param hist_pred: 近期倒数第二次预测值，加载老r、效应系数、新r，但不加载补偿量，任何一维数据，转化为pd.Series，为了作图观察
    :param pred: 未来7天的倒数第二次预测值，加载老r、效应系数、新r，但不加载补偿量，任何一维数据，转化为pd.Series，为了作图观察
    :param period: HoltWinter方法的周期
    :param steps_day: 预测天数
    :return: 加载补偿量的最终预测值，np.array，使其无索引
    """
    if len(pred) < 2:

        return pred - pred
        # raise Exception('pred长度必须≥2')
    steps_day = len(pred)  # 补偿天数
    hist_actual = pd.Series(hist_actual[-29*3:-7])
    hist_pred = pd.Series(hist_pred[-len(hist_actual)-7:-7])
    pred = pd.Series(pred)
    deviation = hist_pred - hist_actual
    weights = []
    for i in range(len(deviation), 0, -1):
        weights.append(i / len(deviation))
    weights = np.array(weights) / sum(weights)


    if len(deviation) <= 3*2:
        logging.warning('historical data are extremely few')
        # LogFile('历史数据过少，不进行偏差量补偿')
        every_compensation = [0] * steps_day
    elif 3*2 < len(deviation) <= 14*2:
        fit_SES = SimpleExpSmoothing(deviation, initialization_method='known', initial_level=np.average(deviation,
            weights=weights)).fit(smoothing_level=1 / 100, optimized=False, use_brute=False, use_boxcox=False, remove_bias=False)
        fit_SES_train = SimpleExpSmoothing(deviation, initialization_method='known', initial_level=np.average(deviation,
            weights=weights)).fit(smoothing_level=1 / 50, optimized=False, use_brute=False, use_boxcox=False, remove_bias=False)


        Holt_add_dam = Holt(deviation, exponential=False, damped_trend=True, initialization_method='known',
                            initial_level=np.average(deviation,
                                                     weights=weights),
                            initial_trend=np.array((sum(deviation[
                                                        int(
                                                            np.ceil(
                                                                len(
                                                                    deviation) / 2)):]) - sum(
                                deviation[:int(np.floor(
                                    len(deviation) / 2))])) / (
                                                       np.floor(
                                                           len(
                                                               deviation) / 2)) ** 2)).fit(smoothing_level=1 / 100,
            smoothing_trend=1 / 100, damping_trend=0.90, optimized=False, use_brute=False,
            use_boxcox=False, remove_bias=False)
        Holt_add_dam_train = Holt(deviation, exponential=False, damped_trend=True, initialization_method='known',
                            initial_level=np.average(deviation,
                                                     weights=weights),
                            initial_trend=np.array((sum(deviation[
                                                        int(
                                                            np.ceil(
                                                                len(
                                                                    deviation) / 2)):]) - sum(
                                deviation[:int(np.floor(
                                    len(deviation) / 2))])) / (
                                                       np.floor(
                                                           len(
                                                               deviation) / 2)) ** 2)).fit(smoothing_level=1 / 50,
                                                                                 smoothing_trend=1 / 50,
                                                                                 damping_trend=0.85, optimized=False,
        use_brute = False, use_boxcox = False, remove_bias = False)

        # 总补偿量为正，则按预测值的正比重新分配每步补偿量，总补偿量为负，则按预测值的反比重新分配每步补偿量；使补偿后的预测值更加稳定，并在多步补偿的情况下与实际值更接近。
        pred = np.array(pred)  # pd.Series是按对应索引进行运算，np.array是按对应顺序进行运算
        ratio_posi = pred / sum(pred)
        ratio_posi[ratio_posi < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_posi = ratio_posi / sum(ratio_posi)
        ratio_neg = 1 / 2 * (1 - ratio_posi) / sum(1 - ratio_posi) + 1 / 2 * (1 / ratio_posi) / sum(1 / ratio_posi)
        ratio_neg[ratio_neg < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_neg = ratio_neg / sum(ratio_neg)
        total_compensation = sum(4 / 5 * np.average(
            np.column_stack([np.array(Holt_add_dam.forecast(steps_day)).reshape(-1, 1), np.array(
                Holt_add_dam_train.forecast(steps_day)).reshape(-1, 1)]), axis=1) + 1 / 5 * np.average(
            np.column_stack([np.array(fit_SES.forecast(steps_day)).reshape(-1, 1), np.array(
                fit_SES_train.forecast(steps_day)).reshape(-1, 1)]), axis=1))
        if total_compensation >= 0:
            every_compensation = total_compensation * ratio_posi
        else:
            every_compensation = total_compensation * ratio_neg

    else:
        Holt_add_dam = Holt(deviation, exponential=False, damped_trend=True, initialization_method='known',
                            initial_level=np.average(deviation,
                                                     weights=weights),
                            initial_trend=np.array((sum(deviation[
                                                        int(
                                                            np.ceil(
                                                                len(
                                                                    deviation) / 2)):]) - sum(
                                deviation[:int(np.floor(
                                    len(deviation) / 2))])) / (
                                                       np.floor(
                                                           len(
                                                               deviation) / 2)) ** 2)).fit(smoothing_level=1 / 100,
            smoothing_trend=1 / 100, damping_trend=0.90, optimized=False, use_brute=False,
            use_boxcox=False, remove_bias=False)
        Holt_add_dam_train = Holt(deviation, exponential=False, damped_trend=True, initialization_method='known',
                            initial_level=np.average(deviation,
                                                     weights=weights),
                            initial_trend=np.array((sum(deviation[
                                                        int(
                                                            np.ceil(
                                                                len(
                                                                    deviation) / 2)):]) - sum(
                                deviation[:int(np.floor(
                                    len(deviation) / 2))])) / (
                                                       np.floor(
                                                           len(
                                                               deviation) / 2)) ** 2)).fit(smoothing_level=1 / 50,
                                                                                 smoothing_trend=1 / 50,
                                                                                 damping_trend=0.85, optimized=False,
        use_brute = False, use_boxcox = False, remove_bias = False)


        HW_add_add_dam = ExponentialSmoothing(deviation, trend='add', damped_trend=True,
                                              seasonal='add', seasonal_periods=period, initialization_method='known',
                                              initial_level=np.average(deviation, weights=weights),
                                              initial_trend=np.array((sum(
                                                  deviation[int(np.ceil(len(deviation) / 2)):]) - sum(
                                                  deviation[:int(np.floor(len(deviation) / 2))])) / (
                                                                         np.floor(len(deviation) / 2)) ** 2),
                                              initial_seasonal=np.array(
                                                  deviation[:len(deviation) // period * period]).reshape(-1,
                                                                                                         period).mean(
                                                  axis=0) - np.average(deviation, weights=weights), use_boxcox=False, missing='raise'). \
            fit(smoothing_level=1 / 100, smoothing_trend=1 / 100, smoothing_seasonal=1 / 100,
                damping_trend=0.90, optimized=False, remove_bias=False, use_brute=False)
        HW_add_add_dam_train = ExponentialSmoothing(deviation, trend='add', damped_trend=True, seasonal='add',
                                                    seasonal_periods=period, initialization_method='known',
                                                    initial_level=np.average(deviation, weights=weights),
                                                    initial_trend=np.array((sum(
                                                        deviation[int(np.ceil(len(deviation) / 2)):]) - sum(
                                                        deviation[:int(np.floor(len(deviation) / 2))])) / (
                                                                               np.floor(len(deviation) / 2)) ** 2),
                                                    initial_seasonal=np.array(
                                                        deviation[:len(deviation) // period * period]).reshape(-1,
                                                                                                               period).mean(
                                                        axis=0) - np.average(deviation, weights=weights), use_boxcox=False, missing='raise'). \
            fit(smoothing_level=1 / 50, smoothing_trend=1 / 50, smoothing_seasonal=1 / 50,
                damping_trend=0.85, optimized=False, remove_bias=False, use_brute=False)

        # 总补偿量为正，则按预测值的正比重新分配每步补偿量，总补偿量为负，则按预测值的反比重新分配每步补偿量；使补偿后的预测值更加稳定，并在多步补偿的情况下与实际值更接近。
        pred = np.array(pred)  # pd.Series是按对应索引进行运算，np.array是按对应顺序进行运算
        ratio_posi = pred / sum(pred)
        ratio_posi[ratio_posi < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_posi = ratio_posi / sum(ratio_posi)
        ratio_neg = 1 / 2 * (1 - ratio_posi) / sum(1 - ratio_posi) + 1 / 2 * (1 / ratio_posi) / sum(1 / ratio_posi)
        ratio_neg[ratio_neg < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_neg = ratio_neg / sum(ratio_neg)
        total_compensation = sum(1 / 2 * np.average(
            np.column_stack([np.array(Holt_add_dam.forecast(steps_day)).reshape(-1, 1), np.array(
                Holt_add_dam_train.forecast(steps_day)).reshape(-1, 1)]), axis=1) + 1 / 2 * np.average(
            np.column_stack([np.array(HW_add_add_dam.forecast(steps_day)).reshape(-1, 1), np.array(
                HW_add_add_dam_train.forecast(steps_day)).reshape(-1, 1)]), axis=1))
        if total_compensation >= 0:
            every_compensation = total_compensation * ratio_posi
        else:
            every_compensation = total_compensation * ratio_neg

    return every_compensation

def compensation_last(hist_actual, hist_pred, pred, period=7):
    """
    :param hist_actual: 近期理论销售历史值，任何一维数据，转化为pd.Series，为了作图观察
    :param hist_pred: 近期倒数第二次预测值，加载老r、效应系数、新r，但不加载补偿量，任何一维数据，转化为pd.Series，为了作图观察
    :param pred: 未来7天的倒数第二次预测值，加载老r、效应系数、新r，但不加载补偿量，任何一维数据，转化为pd.Series，为了作图观察
    :param period: HoltWinter方法的周期
    :param steps_day: 预测天数
    :return: 加载补偿量的最终预测值，np.array，使其无索引
    """
    if len(pred) < 2:

        return pred - pred
        # raise Exception('pred长度必须≥2')
    steps_day = len(pred)  # 补偿天数
    hist_actual = pd.Series(hist_actual[-29*3:])
    hist_pred = pd.Series(hist_pred[-len(hist_actual):])
    pred = pd.Series(pred)
    deviation = hist_pred - hist_actual
    weights = []
    for i in range(len(deviation), 0, -1):
        weights.append(i / len(deviation))
    weights = np.array(weights) / sum(weights)


    if len(deviation) <= 3*2:
        logging.warning('historical data are extremely few')
        # LogFile('历史数据过少，不进行偏差量补偿')
        every_compensation = [0] * steps_day
    elif 3*2 < len(deviation) <= 14*2:
        fit_SES = SimpleExpSmoothing(deviation, initialization_method='known', initial_level=np.average(deviation,
            weights=weights)).fit(smoothing_level=1 / 500, optimized=False, use_brute=False, use_boxcox=False, remove_bias=False)
        fit_SES_train = SimpleExpSmoothing(deviation, initialization_method='known', initial_level=np.average(deviation,
            weights=weights)).fit(smoothing_level=1 / 100, optimized=False, use_brute=False, use_boxcox=False, remove_bias=False)


        Holt_add_dam = Holt(deviation, exponential=False, damped_trend=True, initialization_method='known',
                            initial_level=np.average(deviation,
                                                     weights=weights),
                            initial_trend=np.array((sum(deviation[
                                                        int(
                                                            np.ceil(
                                                                len(
                                                                    deviation) / 2)):]) - sum(
                                deviation[:int(np.floor(
                                    len(deviation) / 2))])) / (
                                                       np.floor(
                                                           len(
                                                               deviation) / 2)) ** 2)).fit(smoothing_level=1 / 500,
            smoothing_trend=1 / 100, damping_trend=0.90, optimized=False, use_brute=False,
            use_boxcox=False, remove_bias=False)
        Holt_add_dam_train = Holt(deviation, exponential=False, damped_trend=True, initialization_method='known',
                            initial_level=np.average(deviation,
                                                     weights=weights),
                            initial_trend=np.array((sum(deviation[
                                                        int(
                                                            np.ceil(
                                                                len(
                                                                    deviation) / 2)):]) - sum(
                                deviation[:int(np.floor(
                                    len(deviation) / 2))])) / (
                                                       np.floor(
                                                           len(
                                                               deviation) / 2)) ** 2)).fit(smoothing_level=1 / 100,
                                                                                 smoothing_trend=1 / 50,
                                                                                 damping_trend=0.85, optimized=False,
        use_brute = False, use_boxcox = False, remove_bias = False)

        # 总补偿量为正，则按预测值的正比重新分配每步补偿量，总补偿量为负，则按预测值的反比重新分配每步补偿量；使补偿后的预测值更加稳定，并在多步补偿的情况下与实际值更接近。
        pred = np.array(pred)  # pd.Series是按对应索引进行运算，np.array是按对应顺序进行运算
        ratio_posi = pred / sum(pred)
        ratio_posi[ratio_posi < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_posi = ratio_posi / sum(ratio_posi)
        ratio_neg = 1 / 2 * (1 - ratio_posi) / sum(1 - ratio_posi) + 1 / 2 * (1 / ratio_posi) / sum(1 / ratio_posi)
        ratio_neg[ratio_neg < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_neg = ratio_neg / sum(ratio_neg)
        total_compensation = sum(4 / 5 * np.average(
            np.column_stack([np.array(Holt_add_dam.forecast(steps_day)).reshape(-1, 1), np.array(
                Holt_add_dam_train.forecast(steps_day)).reshape(-1, 1)]), axis=1) + 1 / 5 * np.average(
            np.column_stack([np.array(fit_SES.forecast(steps_day)).reshape(-1, 1), np.array(
                fit_SES_train.forecast(steps_day)).reshape(-1, 1)]), axis=1))
        if total_compensation >= 0:
            every_compensation = total_compensation * ratio_posi
        else:
            every_compensation = total_compensation * ratio_neg

    else:
        Holt_add_dam = Holt(deviation, exponential=False, damped_trend=True, initialization_method='known',
                            initial_level=np.average(deviation, weights=weights),
                            initial_trend=np.array((sum(deviation[
                                                        int(
                                                            np.ceil(
                                                                len(
                                                                    deviation) / 2)):]) - sum(
                                deviation[:int(np.floor(
                                    len(deviation) / 2))])) / (
                                                       np.floor(
                                                           len(
                                                               deviation) / 2)) ** 2)).fit(smoothing_level=1 / 500,
            smoothing_trend=1 / 100, damping_trend=0.90, optimized=False, use_brute=False,
            use_boxcox=False, remove_bias=False)
        Holt_add_dam_train = Holt(deviation, exponential=False, damped_trend=True, initialization_method='known',
                            initial_level=np.average(deviation,
                                                     weights=weights),
                            initial_trend=np.array((sum(deviation[
                                                        int(
                                                            np.ceil(
                                                                len(
                                                                    deviation) / 2)):]) - sum(
                                deviation[:int(np.floor(
                                    len(deviation) / 2))])) / (
                                                       np.floor(
                                                           len(
                                                               deviation) / 2)) ** 2)).fit(smoothing_level=1 / 100,
                                                                                 smoothing_trend=1 / 50,
                                                                                 damping_trend=0.85, optimized=False,
        use_brute = False, use_boxcox = False, remove_bias = False)


        HW_add_add_dam = ExponentialSmoothing(deviation, trend='add', damped_trend=True,
                                              seasonal='add', seasonal_periods=period, initialization_method='known',
                                              initial_level=np.average(deviation, weights=weights),
                                              initial_trend=np.array((sum(
                                                  deviation[int(np.ceil(len(deviation) / 2)):]) - sum(
                                                  deviation[:int(np.floor(len(deviation) / 2))])) / (
                                                                         np.floor(len(deviation) / 2)) ** 2),
                                              initial_seasonal=np.array(
                                                  deviation[:len(deviation) // period * period]).reshape(-1,
                                                                                                         period).mean(
                                                  axis=0) - np.average(deviation, weights=weights), use_boxcox=False, missing='raise'). \
            fit(smoothing_level=1 / 500, smoothing_trend=1 / 100, smoothing_seasonal=1 / 100,
                damping_trend=0.90, optimized=False, remove_bias=False, use_brute=False)
        HW_add_add_dam_train = ExponentialSmoothing(deviation, trend='add', damped_trend=True, seasonal='add',
                                                    seasonal_periods=period, initialization_method='known',
                                                    initial_level=np.average(deviation, weights=weights),
                                                    initial_trend=np.array((sum(
                                                        deviation[int(np.ceil(len(deviation) / 2)):]) - sum(
                                                        deviation[:int(np.floor(len(deviation) / 2))])) / (
                                                                               np.floor(len(deviation) / 2)) ** 2),
                                                    initial_seasonal=np.array(
                                                        deviation[:len(deviation) // period * period]).reshape(-1,
                                                                                                               period).mean(
                                                        axis=0) - np.average(deviation, weights=weights), use_boxcox=False, missing='raise'). \
            fit(smoothing_level=1 / 100, smoothing_trend=1 / 50, smoothing_seasonal=1 / 50,
                damping_trend=0.85, optimized=False, remove_bias=False, use_brute=False)

        # 总补偿量为正，则按预测值的正比重新分配每步补偿量，总补偿量为负，则按预测值的反比重新分配每步补偿量；使补偿后的预测值更加稳定，并在多步补偿的情况下与实际值更接近。
        pred = np.array(pred)  # pd.Series是按对应索引进行运算，np.array是按对应顺序进行运算
        ratio_posi = pred / sum(pred)
        ratio_posi[ratio_posi < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_posi = ratio_posi / sum(ratio_posi)
        ratio_neg = 1 / 2 * (1 - ratio_posi) / sum(1 - ratio_posi) + 1 / 2 * (1 / ratio_posi) / sum(1 / ratio_posi)
        ratio_neg[ratio_neg < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_neg = ratio_neg / sum(ratio_neg)
        total_compensation = sum(1 / 2 * np.average(
            np.column_stack([np.array(Holt_add_dam.forecast(steps_day)).reshape(-1, 1), np.array(
                Holt_add_dam_train.forecast(steps_day)).reshape(-1, 1)]), axis=1) + 1 / 2 * np.average(
            np.column_stack([np.array(HW_add_add_dam.forecast(steps_day)).reshape(-1, 1), np.array(
                HW_add_add_dam_train.forecast(steps_day)).reshape(-1, 1)]), axis=1))
        if total_compensation >= 0:
            every_compensation = total_compensation * ratio_posi
        else:
            every_compensation = total_compensation * ratio_neg

    return every_compensation