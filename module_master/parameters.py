import numpy as np
import pandas as pd
import math
from yahoofinancials import YahooFinancials


def obtener_datos_historicos(ticker, start_date, end_date):
    yahoo_financials = YahooFinancials(ticker)
    data = yahoo_financials.get_historical_price_data(start_date, end_date, "daily")
    df = pd.DataFrame(data[ticker]["prices"])

    # Convierte la columna de fecha a un tipo de dato datetime
    df['formatted_date'] = pd.to_datetime(df['formatted_date'])

    # Establece la fecha como índice
    df.set_index('formatted_date', inplace=True)

    # Agrupa los datos por mes y calcula la volatilidad mensual
    volatilidad_mensual = df['adjclose'].resample('M').std()

    # Comenzamos a obtener parámetros
    v_0 = volatilidad_mensual.iloc[0]
    theta = volatilidad_mensual.iloc[-1]
    # Retornos logarítmicos para kappa
    ret_log = volatilidad_mensual.pct_change().apply(lambda x: math.log(1 + x)).dropna()
    kappa = ret_log.mean()
    sigmaV = volatilidad_mensual.std()
    # Promedio de precios agrupados por mes de los años requeridos
    proms_adj = df['adjclose'].resample('M').mean()
    # Retornos logarítmicos para correlación de rhoSV
    retlog_prices = proms_adj.pct_change().apply(lambda x: math.log(1 + x)).dropna()
    rhoSV = np.corrcoef(retlog_prices, ret_log)[0,1]

    return df, v_0, theta, kappa, sigmaV, rhoSV



"""
# Convierte la columna de fecha a un tipo de dato datetime
df['formatted_date'] = pd.to_datetime(df['formatted_date'])

# Establece la fecha como índice
df.set_index('formatted_date', inplace=True)

# Agrupa los datos por mes y calcula la volatilidad mensual
volatilidad_mensual = df['adjclose'].resample('M').std()
#print(volatilidad_mensual)

# Comenzamos a obtener párametros
v_0 = volatilidad_mensual.iloc[0]
theta = volatilidad_mensual.iloc[-1]
#Retornos logaritmicos para kappa
ret_log = volatilidad_mensual.pct_change().apply(lambda x: math.log(1 + x)).dropna()
kappa = ret_log.mean()
sigmaV = volatilidad_mensual.std()
#Promedio de precios agrupados por mes de los años requeridos
proms_adj = df['adjclose'].resample('M').mean()
#Retornos logaritmicos para correlacion de rhoSV
retlog_prices = proms_adj.pct_change().apply(lambda x: math.log(1 + x)).dropna()
rhoSV = np.corrcoef(retlog_prices, ret_log)[0,1]
#print(v_0,theta,kappa, sigmaV, rhoSV)
"""
