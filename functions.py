import pandas as pd
from pathlib import Path
# import numpy as np
# from decorators import timer
# import time


ticks = [["ETHUSDT", 2], ["BATUSDT", 4], ["MKRUSDT", 1], ["FLMUSDT", 4],
         ["ANKRUSDT", 5], ["MTLUSDT", 4], ["CTSIUSDT", 4],
         ["1000LUNCUSDT", 7], ["CFXUSDT", 7], ["XVSUSDT", 6]
         ]


class Bars:
    def __init__(self, ticker):
        self.ticker = ticker

    def get_bars(self, minutes, days, is_list=False):
        """Retrieve historical bars from a pickle file."""
        # Get the absolute path of the current script
        script_path = Path(__file__).resolve()

        # Find the root directory (parent of the system folder)
        root_dir = script_path.parent.parent

        # Construct the bars folder path dynamically
        bars_dir = root_dir / "bars"

        # Ensure bars directory exists
        if not bars_dir.exists():
            raise FileNotFoundError(f"Bars directory not found: {bars_dir}")

        # Construct file name based on the timeframe
        file_name = f"{minutes}{self.ticker}.pkl" if minutes > 1 else f"{self.ticker}.pkl"
        file_path = bars_dir / file_name
        try:
            df = pd.read_pickle(file_path)
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            df['fecha'] = df['date']
            regs = int(days * (1440 / minutes))
            df_sliced = df[-regs:]
            # index need a reset
            df_sliced = df_sliced.reset_index()
            if is_list:
                df_sliced = df_sliced.to_dict(orient='records')
            return df_sliced
        except Exception as e:
            raise ValueError(f"an errror was commited: {e}")


'''
def filtered_df(df, size1, size2, size3):
    # 3bp, have size of 1st bar, precentage of size of second bar (or lower)
    # and percentage of third bar (bar1 perecentage or higher)
    df = df.reset_index(drop=True)
    df['index'] = df.index
    # in this case, we nned first the color
    df['side'] = np.where(df['close'] > df['open'], "BUY", "SELL")
    # now the body sie
    df['size'] = abs((df['open'] - df['close']) / df['open'])
    # we need the hour too
    df['hour'] = df['date'].dt.hour
    # let's filter it
    filter_df = df.loc[
        (df['size'].shift(2) > size1)
        & (df['size'].shift(1) < df['size'].shift(2) * size2)
        & (df['size'] > df['size'].shift(2) * size3)
        & ((df['side'].shift(1) != df['side'].shift(2))
           & (df['side'] != df['side'].shift(1)))
        & (df['hour'] != 26)
        ]
    return filter_df


def single_bet(ticker, df, dfMin, minutes, bet, TP,
               SL, bars, double=False, distance=0, 
               getIns=False, reverse=False):
    ret = []
    data = 0
    sl = 0.0125
    for row in df.itertuples(index=True):
        if data != 0:  # the first one don't contain any info
            if data['dateOut'] >= row.date:  # the entries are too close
                continue
        if reverse:
            if row.side == 'BUY':
                side = 'SELL'
            else:  # that means, side="SELL"
                side = "BUY"
        else:
            side = row.side
        priceIn = row.close
        dateIn = row.date
        Break = {
            'breakPrice': priceIn,
            'breakDate': dateIn,
            'side': side,
            'size': row.size,  # the "C" bars, shows the path
            'pointer': row.index,
        }
        pointer = (row.index + 1) * minutes
        slice = dfMin[pointer:]
        # double marks the direction
        if double:
            # first, need the side and the bars "consumed" to added into pointer, and refresh the slice, and priceIn
            data = doubleBet(ticker, distance, slice, priceIn)
            if data == 0:  # it's over
                break
            side = data['side']
            pointer += data['pointer']
            slice = dfMin[pointer + 1:]
            priceIn = data['openPrice']
            dateIn = data['openDate']
        data = directBet(ticker, bet, side, slice, priceIn)
        if data != 0:
            # now if the original bet loss, let's protect it!!
            if data['outcome'] == 'tp':
                # well, if double, the priceIn is precisly, the actual priceIn, also dateIn
                if double:
                    data['priceIn'] = priceIn
                    data['dateIn'] = dateIn
                else:
                    data['priceIn'] = Break['breakPrice']
                    data['dateIn'] = Break['breakDate']
                data['breakPrice'] = Break['breakPrice']
                data['breakDate'] = Break['breakDate']
                data['ticker'] = ticker
                data['size'] = Break['size']
                ret.append(data)
            else:
                init = data['pointer'] + pointer + 1  # the bars "consumed" by the system
                end = init + bars  # just 1440 bars
                slice = dfMin[init:end]
                if data['side'] == 'SELL':
                    side = "BUY"
                    entradaBuy = data['priceOut']
                    entradaSell = 0
                else:
                    side = "SELL"
                    entradaSell = data['priceOut']
                    entradaBuy = 0
                # checarEntrada(ticker, lista, sl, slMax, tp, elajuste, posi, entradabuy,
                # entradasell, lafechaIn, p, eldeta, punteroIn)
                data = checarEntrada(
                    ticker, slice, sl, SL, TP, data['adjust'], side,
                    entradaBuy, entradaSell, data['dateOut'], data['pointer'], data, pointer
                )
                if data != 0:
                    # well, if double, the priceIn is precisely, the actual priceIn, also dateIn
                    if double:
                        data['priceIn'] = priceIn
                        data['dateIn'] = dateIn
                    else:
                        data['priceIn'] = Break['breakPrice']
                        data['dateIn'] = Break['breakDate']
                    data['breakPrice'] = Break['breakPrice']
                    data['breakDate'] = Break['breakDate']
                    data['ticker'] = ticker
                    data['size'] = Break['size']
                    ret.append(data)
    records = len(ret)
    if records == 0:
        return 0
    df = pd.DataFrame(ret)
    Sum = df['profit'].sum()
    positives = df.loc[df['profit'] > 0]['profit'].count() / records
    directs = df.loc[df['way'] == 'direct']['way'].count() / records
    grouped = pd.DataFrame(df.groupby(['way', 'outcome']).size())
    sls = df.loc[df['outcome'] == 'sl']['outcome'].count() / records
    if getIns:
        Exit = {
            'ticker': ticker,
            'sum': Sum,
            'positives': positives,
            'records': records,
            'directs': directs,
            'sl': sls,
            'grouped': grouped,
            'bet': bet,
            'distance': distance,
            'TP': TP,
            'SL': SL,
            'bars': bars,
            'getIns': df,
        }
    else:
        Exit = {
            'ticker': ticker,
            'sum': Sum,
            'positives': positives,
            'records': records,
            'directs': directs,
            'sl': sls,
            'grouped': grouped,
            'bet': bet,
            'distance': distance,
            'TP': TP,
            'SL': SL,
            'bars': bars,
        }
    return Exit


# now a function that can review if was touched tp or sl first


def checarEntrada(
        ticker, lista, sl, slMax, tp, elajuste, posi, entradabuy,
        entradasell, lafechaIn, p, eldeta, punteroIn):
    # por ser diferentes parámetros, se saca del master
    posicion = posi
    inbuy = entradabuy
    insell = entradasell
    ajusteloss = elajuste
    fechaIn = lafechaIn
    ganancia = 0
    # TODO se hará con while, so, o se acaba la lista, se va a SLfinal o a TP
    pI = 0
    empate = True
    detalle = [eldeta]  # list starts with the first element
    dato = {}
    eltp = 0
    elsl = 0
    for m in range(len(lista)):
        # existen tres escenarios, se va a SL, donde vuelve aquó con pos, ajusteloss y demás, se va a TP donde sale
        # o se termina la barra donde sale también
        ellow = lista[m]['low']
        elhigh = lista[m]['high']
        if posicion == "BUY":  # inicia en largos el sistema, ponemos entrada con sl (condicion1)
            elsl = inbuy * (1 - sl)
            eltp = inbuy * (1 + tp + ajusteloss)
            eltp = round(eltp, ticker[1])
            elsl = round(elsl, ticker[1])
            if ellow <= elsl:  # OJO ajustamos loss y "cambiamos" de posicion
                # antes que nada, el dato se tiene que ir a detalle
                fechaOut = lista[m]['fecha']
                ajusteloss += sl + 0.0008  # por la comision
                # OJO, si el máximo pasa de 10% nos vamos
                if ajusteloss >= slMax:  # bay bay caiman
                    profit = -ajusteloss
                    dato = {
                        'side': posicion,
                        'dateOut': fechaOut,
                        'priceOut': elsl,
                        'punteroIn': punteroIn,
                        'punteroOut': punteroIn + p + pI,
                        'profit': profit,
                        'ajusteloss': ajusteloss,
                        'sl': elsl,
                        'tp': eltp,
                        'outcome': "sl",
                        'way': 'indirect',
                        'detalle': detalle,
                        'pointer': p + pI,  # sería el puntero definitivo en la lista
                    }
                    empate = False
                    break
                dato = {
                    'ticker': ticker,
                    'posicion': posicion,
                    'fechaIn': fechaIn,
                    'precioIn': inbuy,
                    'fechaOut': fechaOut,
                    'precioOut': elsl,
                    'punteroIn': punteroIn,
                    'punteroOut': punteroIn + p + pI,
                    'ganancia': -sl - 0.0008,
                    'ajusteloss': ajusteloss,
                    'sl': elsl,
                    'tp': eltp,
                    'resultado': "sl",
                    'forma': 'indirecta',
                    'puntero': p + pI,  # sería el puntero definitivo en la lista
                }
                detalle.append(dato)
                fechaIn = lista[m]['fecha']  # en cada iteracción agrega a detalle cuando "entró"
                posicion = "SELL"
                insell = elsl  # por que en la siguiente interacción entraría en este nuevo "open"
            elif elhigh >= eltp:  # ganamos sres
                fechaOut = lista[m]['fecha']
                # calculate profit based on RSI
                profit = tp - 0.0008
                dato = {
                    'side': posicion,
                    'dateOut': fechaOut,
                    'punteroIn': punteroIn,
                    'punteroOut': punteroIn + p + pI,
                    'priceOut': eltp,
                    'profit': profit,
                    'ajusteloss': ajusteloss,
                    'sl': elsl,
                    'tp': eltp,
                    'outcome': "tp",
                    'way': 'indirect',
                    'detalle': detalle,
                    'pointer': p + pI,  # sería el puntero definitivo en la lista
                }
                empate = False
                break
        elif posicion == "SELL":  # inicia en cortos el sistema, ponemos entrada con sl (condicion1)
            elsl = insell * (1 + sl)
            elsl = round(elsl, ticker[1])
            eltp = insell * (1 - ajusteloss - tp)
            eltp = round(eltp, ticker[1])
            if elhigh >= elsl:  # OJO ajustamos loss y "cambiamos" de posicion
                ajusteloss += sl + 0.0008  # por la comision
                # OJO, si el máximo pasa de 10% nos vamos
                # antes que nada, el dato se tiene que ir a detalle
                fechaOut = lista[m]['fecha']
                if ajusteloss >= slMax:  # bay bay caiman
                    profit = -ajusteloss
                    dato = {
                        'side': posicion,
                        'dateOut': fechaOut,
                        'priceOut': elsl,
                        'punteroIn': punteroIn,
                        'punteroOut': punteroIn + p + pI,
                        'profit': profit,
                        'ajusteloss': ajusteloss,
                        'sl': elsl,
                        'tp': eltp,
                        'outcome': "sl",
                        'way': 'indirect',
                        'detalle': detalle,
                        'pointer': p + pI,  # sería el puntero definitivo en la lista
                    }
                    empate = False
                    break
                dato = {
                    'ticker': ticker,
                    'posicion': posicion,
                    'fechaIn': fechaIn,
                    'precioIn': insell,
                    'fechaOut': fechaOut,
                    'precioOut': elsl,
                    'punteroIn': punteroIn,
                    'punteroOut': punteroIn + p + pI,
                    'ganancia': -sl - 0.0008,
                    'ajusteloss': ajusteloss,
                    'sl': elsl,
                    'tp': eltp,
                    'resultado': "sl",
                    'forma': 'indirecta',
                    'puntero': p + pI,  # sería el puntero definitivo en la lista
                }
                detalle.append(dato)
                fechaIn = lista[m]['fecha']  # en cada iteracción agrega a detalle cuando "entró"
                posicion = "BUY"
                inbuy = elsl  # por que en la siguiente interacción entraría en este nuevo "open"
            elif ellow <= eltp:  # ganamos sres
                fechaOut = lista[m]['fecha']
                profit = tp - 0.0008
                # OJO en el 5 de misbarras esta el epoch
                dato = {
                    'side': posicion,
                    'dateOut': fechaOut,
                    'priceOut': eltp,
                    'punteroIn': punteroIn,
                    'punteroOut': punteroIn + p + pI,
                    'profit': profit,
                    'ajusteloss': ajusteloss,
                    'sl': elsl,
                    'tp': eltp,
                    'outcome': "tp",
                    'way': 'indirect',
                    'detalle': detalle,
                    'pointer': p + pI,  # sería el puntero definitivo en la lista
                }
                empate = False
                break
        pI += 1
    if empate:
        # recordar que m contiene el último registro
        fechaOut = lista[-1]['fecha']
        elclose = lista[-1]['close']
        if posicion == "BUY":
            ganancia = ((elclose - inbuy) / inbuy) - 0.0008 - ajusteloss
        elif posicion == "SELL":
            ganancia = ((insell - elclose) / insell) - 0.0008 - ajusteloss
        profit = ganancia
        dato = {
            'side': posicion,
            'dateOut': fechaOut,
            'priceOut': elclose,
            'punteroIn': punteroIn,
            'punteroOut': punteroIn + p + pI,
            'profit': profit,
            'ajusteloss': ajusteloss,
            'sl': elsl,
            'tp': eltp,
            'outcome': "tie",
            'way': 'indirect',
            'detalle': detalle,
            'pointer': p + pI,  # sería el puntero definitivo en la lista
        }

    return dato


def doubleBet(ticker, distance, listMin, priceIn):
    data = 0  # just for case that no one in reach the threshold
    # seek until price ouch one of both prices
    in_buy, in_sell = round(
        priceIn * (1 + distance), ticker[1]), round(
        priceIn * (1 - distance), ticker[1]
    )
    for pI in range(len(listMin)):
        if listMin[pI]['high'] >= in_buy:
            data = {
                'openDate': listMin[pI]['fecha'],
                'openPrice': in_buy,
                'side': "BUY",
                'inBuy': in_buy,
                'inSell': in_sell,
                'pointer': pI  # the bars "consumed" in this function
            }
            return data
        elif listMin[pI]['low'] <= in_sell:
            data = {
                'openDate': listMin[pI]['fecha'],
                'openPrice': in_sell,
                'side': "SELL",
                'inBuy': in_buy,
                'inSell': in_sell,
                'pointer': pI  # the bars "consumed" in this function
            }
            return data
    return data


def directBet(ticker, bet, side, listaMin, precioIn):
    # ticker is a list, the first value have the ticker name, the second one, prescition of ticker
    data = 0
    # get the bet
    if side == "BUY":
        tp = precioIn * (1 + bet)
        sl = precioIn * (1 - bet)
        tp = round(tp, ticker[1])
        sl = round(sl, ticker[1])
        for pI in range(len(listaMin)):
            if listaMin[pI]['high'] >= tp:  # winner winner chicken dinner
                fechaOut = listaMin[pI]['fecha']
                ganancia = bet - 0.0008
                data = {
                    'side': side,
                    'dateOut': fechaOut,
                    'priceOut': tp,
                    'profit': ganancia,
                    'sl': sl,
                    'tp': tp,
                    'outcome': 'tp',
                    'way': 'direct',
                    'pointer': pI,  # the bars "consumed" in this function
                }
                return data
            elif listaMin[pI]['low'] <= sl:
                if pI + 1440 >= len(listaMin):  # nos vamos
                    break
                fechaOut = listaMin[pI]['fecha']
                ajuste = bet + 0.0008
                data = {
                    'side': side,
                    'dateOut': fechaOut,
                    'priceOut': sl,
                    'profit': -ajuste,
                    'sl': sl,
                    'tp': tp,
                    'outcome': 'sl',
                    'way': 'direct',
                    'adjust': ajuste,
                    'pointer': pI,  # the bars "consumed" in this function
                }
                return data
    elif side == "SELL":
        tp = precioIn * (1 - bet)
        sl = precioIn * (1 + bet)
        tp = round(tp, ticker[1])
        sl = round(sl, ticker[1])
        for pI in range(len(listaMin)):
            if listaMin[pI]['low'] <= tp:  # winner winner chicke
                fechaOut = listaMin[pI]['fecha']
                ganancia = bet - 0.0008
                data = {
                    'side': side,
                    'dateOut': fechaOut,
                    'priceOut': tp,
                    'profit': ganancia,
                    'sl': sl,
                    'tp': tp,
                    'outcome': 'tp',
                    'way': 'direct',
                    'pointer': pI,  # the bars "consumed" in this function
                }
                return data
            elif listaMin[pI]['high'] >= sl:
                # por el momento usaremos sl en 0.01 y tp en 0.1, pero luego se puede modificar
                fechaOut = listaMin[pI]['fecha']
                ajuste = bet + 0.0008
                data = {
                    'side': side,
                    'dateOut': fechaOut,
                    'priceOut': sl,
                    'profit': -ajuste,
                    'sl': sl,
                    'tp': tp,
                    'outcome': 'sl',
                    'way': 'direct',
                    'adjust': ajuste,
                    'pointer': pI,  # son las barras que duró el sistema
                }
                return data
    # just for the case where no one bet is reached
    return data


@timer
def analisys(ticker, df, dfMin, minutes, reverse):
    # ok the args takes the ranges for analisys, the kwargs takes the key
    maximum = - 1000
    alls = []
    winner = []
    # parameter sizeB and sizeC are fixed
    sizeC = 0.4
    bars = 1440
    # the very first thing to do, is get the si
    # ze, after that, the bet
    for sizeA in np.arange(0.01, 0.0201, 0.005):  # 4 iter
        for sizeB in np.arange(0.2, 0.61, 0.2):
            f = filtered_df(df, sizeA, sizeB, sizeC)
            for bet in np.arange(0.02, 0.041, 0.01):  # 3 iter 4*5*3=60
                for SL in np.arange(0.04, 0.11, 0.02):  # takes 5 iteractions
                    for TP in np.arange(0.06, 0.11, 0.02):  # ok takes 3 iteractions
                        In = single_bet(
                            ticker=ticker, df=f, dfMin=dfMin, minutes=minutes,
                            bet=bet, getIns=False, reverse=reverse, SL=SL, TP=TP, bars=bars
                        )
                        if In != 0:
                            if In['sum'] > maximum:
                                winner = In
                                maximum = In['sum']
                            In['sizeA'] = sizeA,
                            In['sizeB'] = sizeB,
                            In['sizeC'] = sizeC,
                            alls.append(In)
                        else:
                            continue
                        del In  # to improve memory management
    alls = pd.DataFrame(alls)
    ret = {
        'ticker': ticker,
        'winner': winner,
        'alls': alls
    }
    return ret



def analisys_mp(ticker, df, dfMin, minutes, reverse):
    winner = []
    sizeC = 0.4
    bars = 1440
    process1 = []
    process2 = []
    for sizeA in np.arange(0.01, 0.0201, 0.005):  # 4 iterations
        for sizeB in np.arange(0.2, 0.61, 0.2):  # 3 iterations
            f = filtered_df(df, sizeA, sizeB, sizeC)
            for bet in np.arange(0.02, 0.041, 0.01):  # 3 iterations
                for SL in np.arange(0.04, 0.11, 0.02):  # 4 iterations
                    # instead the last for, directly the process
                    args1 = {'ticker': ticker, 'df': f, 'dfMin': dfMin, 'minutes': minutes,
                        'bet': bet, 'getIns': False, 'reverse':reverse,
                        'SL': SL, 'TP': 0.04, 'bars': bars
                        }
                    with ProcessPoolExecutor() as executor:
                        future = executor.submit(single_bet, **args1)  # Submit task
                        result = future.result()  # Wait for and retrieve result
                    process1.append(result)
                    args2 = {'ticker': ticker, 'df': f, 'dfMin': dfMin, 'minutes': minutes,
                             'bet': bet, 'getIns': False, 'reverse': reverse,
                             'SL': SL, 'TP': 0.06, 'bars': bars
                             }
                    with ProcessPoolExecutor() as executor:
                        future = executor.submit(single_bet, **args2)  # Submit task
                        result = future.result()  # Wait for and retrieve result
                    process2.append(result)
    alls = list(process1) + list(process2)
    if alls:
        df_alls = pd.DataFrame(alls)
        winner = df_alls.loc[df_alls['sum'].idxmax()].to_dict()

    ret = {
        'ticker': ticker,
        'winner': winner,
        'alls': alls
    }
    return ret


@timer
def analisys_double(ticker, df, dfMin, minutes):
    maximum = -1000
    alls = []
    winner = []
    for gap in np.arange(0.02, 0.11, 0.02):  # 4 iter
        f = filter(df, gap)
        for bet in np.arange(0.03, 0.051, 0.01):  # 3 iter
            for distance in np.arange(0.01, 0.031, 0.005):  # 5 iter
                # 4*5*3*5=300, aprox 5 mins per ticker
                In = single_bet(ticker=ticker, df=f, dfMin=dfMin, minutes=minutes,
                                bet=bet, getIns=False, distance=distance, double=True)
                if In != 0:
                    In['gap'] = gap
                    if In['sum'] > maximum:
                        winner = In
                        maximum = In['sum']
                    alls.append(In)
                else:
                    continue
                del In  # to improve memory management
    alls = pd.DataFrame(alls)
    ret = {
        'ticker': ticker,
        'winner': winner,
        'alls': alls
    }
    return ret


@timer
def ana10(minutes, reverse):
    alls = []
    days = 365
    for a in ticks:
        smb = a[0]
        df = getbars(smb, minutes, days)
        dfMin = getbars(smb, 1, days, l=True)
        ana = analisys(a, df, dfMin, minutes, reverse)
        alls.append(ana['alls'])
        print(smb, ana['winner']['sum'], time.ctime())
        # save for avoid dataloss
        if not reverse:
            name = "temp/" + str(minutes) + smb + ".pkl"
        else:
            name = "temp/" + str(minutes) + smb + "r" + ".pkl"
        with open(name, "wb") as f:
            pk.dump(ana, f)

    # make it DF
    ret = pd.concat(alls)
    try:
        grouped = ret.groupby(
            ['sizeA', 'sizeB', "sizeC", 'SL', 'TP', 'bars', 'bet'])[
            ['sum', 'positives', 'directs', 'records', 'sl']].mean()
    except Exception as e:
        print(e)
        grouped = 0
    Exit = {
        'noGrouped': ret,
        'grouped': grouped,
    }
    # finally, save the complete file
    if not reverse:
        name = "pkl/ana10" + str(minutes) + ".pkl"
    else:
        name = "pkl/ana10r" + str(minutes) + ".pkl"
    with open(name, "wb") as f:
        pk.dump(Exit, f)
    return Exit


@timer
def ana10_review(mins, gap, bet, reverse=False, double=False, distance=0):
    al = []
    for t in ticks:
        df = getbars(t[0], mins, 365)
        f = filter(df, gap)
        dfMin = getbars(t[0], 1, 365, True)
        if not double:
            s = single_bet(t, f, dfMin, mins, bet, reverse=reverse)
        else:
            s = single_bet(t, f, dfMin, mins, bet, reverse=reverse, double=double, distance=distance)
        al.append(s)
    # get the statistics
    df = pd.DataFrame(al)
    positives = df.loc[df['sum'] > 0]['sum'].count()
    avg = df['sum'].mean()
    sd = df['sum'].std()
    ret = {
        'positives': positives,
        'avg': avg,
        'sd': sd,
        'records': df,
    }
    return ret


@timer
def ana10_double(minutes):
    alls = []
    days = 365
    for a in ticks:
        smb = a[0]
        df = getbars(smb, minutes, days)
        dfMin = getbars(smb, 1, days, l=True)
        ana = analisys_double(a, df, dfMin, minutes)
        alls.append(ana['alls'])
        print(smb, ana['winner']['sum'], time.ctime())
        # save for avoid dataloss
        name = "temp/aD" + str(minutes) + smb + ".pkl"
        with open(name, "wb") as f:
            pk.dump(ana, f)

    # make it DF
    ret = pd.concat(alls)
    try:
        grouped = ret.groupby(['gap', 'distance', 'bet'])[['sum', 'positives', 'directs', 'records', 'sl']].mean()
    except Exception as e:
        print(e)
        grouped = 0
    Exit = {
        'noGrouped': ret,
        'grouped': grouped,
    }
    # finally, save the complete file
    name = "pkl/ana10D" + str(minutes) + ".pkl"
    with open(name, "wb") as f:
        pk.dump(Exit, f)
    return Exit


def tickers():
    with open("tickers.csv", 'r') as csv_file:
        csv_reader = reader(csv_file)
        # Passing the cav_reader object to list() to get a list of lists
        salida = list(csv_reader)
    return salida


@timer
def alls(minutes, size1, size2, size3, bet, SL, TP, bars, distance=0, reverse=False, double=False):
    params = locals().copy()
    all = []
    recs = []
    days = 365
    p = 0
    for a in tickers():
        # if the ticker doesn't exist or have an error, will raise an error
        try:
            listMin = getbars(a[0], 1, days, l=True)
            lisT = getbars(a[0], minutes, days)
            tk = a[0], int(a[1])
            f = filter(lisT, size1, size2, size3)
            if double:
                e = single_bet(ticker=tk, dfMin=listMin, df=f, bet=bet, distance=distance, getIns=True, reverse=reverse,
                               minutes=minutes, double=True)
            else:
                e = single_bet(ticker=tk, dfMin=listMin, df=f, bet=bet, getIns=True, reverse=reverse, minutes=minutes,
                               SL=SL, TP=TP, bars=bars)
            if e != 0:
                all.append(e)
                r = e['getIns']
                recs.append(r)
            else:
                print(a, "doesn't have any records")
        except Exception as e:
            print(a, f"raise an error {e}")
        if p % 50 == 0:
            print(p, "records", time.ctime())
        p += 1

    # make it pandas
    df = pd.DataFrame(all)
    records = pd.concat(recs)
    # get the averages, alls, and positives, and top10
    try:
        avgs_all = df[['sum', 'positives', 'directs', 'records', 'sl']].mean()
        positives = (df.loc[df['sum'] > 0]['sum'].count()) / len(df)
        avg_top10 = df.sort_values(by='sum', ascending=False).head(10)[
            ['sum', 'positives', 'directs', 'records', 'sl']].mean()
        top10 = df.sort_values(by='sum', ascending=False).head(10)[
            ['ticker', 'sum', 'positives', 'directs', 'records', 'sl']]
        ret = {
            'params': params,
            'alls': df,
            'records': records,
            'avg_alls': avgs_all,
            'positives': positives,
            'avg_top10': avg_top10,
            'top10': top10,
        }
    except Exception as e:
        print(f"there's some issues in total section {e}")
        ret = {
            'params': params,
            'alls': df,
            'records': records,
        }
    if not reverse and not double:
        name = "pkl/all" + str(minutes) + ".pkl"
    elif reverse and not double:
        name = "pkl/allr" + str(minutes) + ".pkl"
    else:
        name = "pkl/alld" + str(minutes) + ".pkl"
    with open(name, "wb") as f:
        pk.dump(ret, f)
    return ret


def reach_max_loss(data):
    d = pd.DataFrame(data)
    maximus = []
    max_bal = d.iloc[0]['balance']  # the first record
    data = (0, max_bal)
    maximus.append(data)
    for i in range(1, len(d)):
        bal = d.iloc[i]['balance']
        if bal > max_bal:
            data = (i, bal)
            maximus.append(data)
            max_bal = bal
    MTOne = []
    for a in range(1, len(maximus)):
        iprev = maximus[a - 1][0]
        i = maximus[a][0]
        size = i - iprev
        if size > 1:
            data = iprev, i
            MTOne.append(data)
    mins = []
    for b in MTOne:
        lower = d[b[0]:b[1]]['balance'].min()
        change = (lower - d.iloc[b[0]]['balance']) / d.iloc[b[0]]['balance']
        mins.append(change)
    cols = ["min"]
    dataF = pd.DataFrame(mins, columns=cols)
    daMin = dataF['min'].min()
    return daMin


def compoundCapital(entradas, porIn, saldoI, getIns=False):
    # las entradas son en pandas, el lvg siempre es 20
    saldo = saldoI
    minimo = 100000000
    maximo = -100000000
    acum = []
    # ok, the data out is profit, money_in, balance in every row
    for a in range(len(entradas)):
        entrada = saldo * porIn
        ganancia = entrada * 20 * entradas.iloc[a]['profit']
        saldo += ganancia
        # get data
        data = {
            'ticker': entradas.iloc[a]['ticker'],
            'dateIn': entradas.iloc[a]['dateIn'],
            'dateOut': entradas.iloc[a]['dateOut'],
            'money': entrada,
            'outcome': entradas.iloc[a]['outcome'],
            'way': entradas.iloc[a]['way'],
            'win_rate': entradas.iloc[a]['profit'],
            'profit': ganancia,
            'balance': saldo,
        }
        acum.append(data)
        if saldo < minimo:
            minimo = saldo
        if saldo > maximo:
            maximo = saldo
    # retornamos los valores minimos y maximos
    # ok now wee need to calculate the maximum percentage of loss in every cicle taht reach a new maximum
    daMin = reach_max_loss(acum)
    if getIns:
        fin = {
            'ticker': entradas.iloc[0]['ticker'][0],
            'min': round(minimo, 2),
            'max': round(maximo, 2),
            'finalB': round(saldo, 2),
            'max_loss': round(daMin, 3),
            'records': acum,
        }
    else:
        fin = {
            'ticker': entradas.iloc[0]['ticker'][0],
            'min': round(minimo, 2),
            'max': round(maximo, 2),
            'finalB': round(saldo, 2),
            'max_loss': round(daMin, 3),
        }
    return fin


@timer
def make_df_list(t: list, gap: float, bet: float):
    recs = []
    for tk in t:
        df = getbars(tk[0], 60, 365)
        f = filter(df, gap)
        dfMin = getbars(tk[0], 1, 365, True)
        s = single_bet(tk, f, dfMin, 60, bet, getIns=True)
        r = s['getIns']
        recs.append(r)
    final = pd.concat(recs)
    return final


# filter take the first row with the criteria, and if there's any date repeated, the filter removes them


def filter_df(df):
    # the very fisrt thing, is sort it
    df = df.sort_values(['dateIn', 'gap'], ascending=[True, True])
    df_final = df.drop_duplicates(subset='dateIn', keep='first').reset_index(drop=True)
    df_final = df_final.reset_index(drop=True)
    return df_final


# clean df, remove the "solaped" records, it means, that none
# dateIn in next record is greatther than dateOut of previews


def clean_df(df):
    # this function clean the overlaped dates between records
    borrables = []
    q = 0
    for p in range(1, len(df)):
        if p < q:
            continue
        # si es menor la fechaIn que la fechaOut, la agregamos, pero nos "metemos" a otro contador para sacar a todos
        if df.iloc[p]['dateIn'] < df.iloc[p - 1]['dateOut']:
            # nos metemos a otro contador
            for q in range(p, len(df)):
                if df.iloc[q]['dateIn'] < df.iloc[p - 1]['dateOut']:
                    borrables.append(q)
                else:
                    q += 1  # para que no cicle en el último registro
                    break
    # let's return the list of cleaned and the leftover
    cleaned = df.drop(index=borrables).reset_index(drop=True)
    leftover = df.iloc[borrables]
    # reset the index to be re-usable
    cleaned = cleaned.reset_index(drop=True)
    leftover = leftover.reset_index(drop=True)

    return {'cleaned': cleaned, 'leftover': leftover}
'''
