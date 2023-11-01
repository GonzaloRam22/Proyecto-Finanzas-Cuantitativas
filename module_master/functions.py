import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from scipy.optimize import minimize_scalar


def generate_heston_paths(S, T, r, kappa, theta, v_0, rho, xi,
                          steps, Npaths, return_vol=False):
    dt = T / steps
    size = (Npaths, steps)
    prices = np.zeros(size)
    sigs = np.zeros(size)
    S_t = S
    v_t = v_0
    for t in range(steps):
        WT = np.random.multivariate_normal(np.array([0, 0]),
                                           cov=np.array([[1, rho],
                                                         [rho, 1]]),
                                           size=Npaths) * np.sqrt(dt)

        S_t = S_t * (np.exp((r - 0.5 * v_t) * dt + np.sqrt(v_t) * WT[:, 0]))
        v_t = np.abs(v_t + kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * WT[:, 1])
        prices[:, t] = S_t
        sigs[:, t] = v_t

    if return_vol:
        return prices, sigs

    return prices

"""
kappa = kappa
theta = theta
v_0 =  v_0
xi = sigmaV
r = 0.0538 #tomada de https://www.bloomberg.com/markets/rates-bonds/government-bonds/us a un año
S = df["adjclose"].iloc[-1]
Npaths = 5000
steps = 252
T = 1
rho = rhoSV
"""

def graficar_simulacion_precios_y_volatilidades(prices, volatilities, steps):
    # Gráfica de precios
    plt.figure(figsize=(12, 5))  # Opcional: ajusta el tamaño de la figura
    plt.subplot(1, 2, 1)  # Divide la figura en 1 fila y 2 columnas, selecciona la primera subtrama
    for i in range(steps):
        plt.plot(prices[i, :])
    plt.title('Heston Prices Paths Simulation')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.axis([0, 250, 0, 3000])

    # Gráfica de volatilidades
    plt.subplot(1, 2, 2)  # Selecciona la segunda subtrama
    for i in range(steps):
        plt.plot(volatilities[i, :])
    plt.title('Heston Stochastic Volatility Simulation')
    plt.xlabel('Time Steps')
    plt.ylabel('Volatility')

    plt.tight_layout()  # Ajusta automáticamente los márgenes para evitar solapamientos
    plt.show()

def graficar_simulacion_precio_y_volatilidad2(prices, volatilities, steps, y, theta):
    plt.figure(figsize=(14, 6))  # Ajusta el tamaño de la figura

    # Gráfica de precios
    plt.subplot(1, 2, 1)  # Divide la figura en 1 fila y 2 columnas, selecciona la primera subtrama
    num_trayectorias= 10
    for i in range(num_trayectorias):
        plt.plot(prices[i])
    plt.title('Heston Price Paths Simulation')
    plt.axhline(y, color='blue', label='Strike Price Mean')
    plt.axis([0, 250, 0, 600])
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.legend(fontsize=15)

    # Gráfica de volatilidades
    plt.subplot(1, 2, 2)  # Selecciona la segunda subtrama
    for i in range(num_trayectorias):
        plt.plot(volatilities[i])
    plt.axhline(np.sqrt(theta), color='black', label=r'$\sqrt{\theta}$')
    plt.title('Heston Stochastic Vol Simulation')
    plt.xlabel('Time Steps')
    plt.ylabel('Volatility')
    plt.legend(fontsize=15)

    plt.tight_layout()  # Ajusta automáticamente los márgenes para evitar solapamientos
    plt.show()


#Funcion para calcular los puts y calls de las opciones
def calculate_option_prices(S_t, K, T, r, sigma, type_='call'):
    d1 = (np.log(S_t / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    

    if type_ == 'call':
        option_price = S_t * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif type_ == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S_t * norm.cdf(-d1)
    else:
        raise ValueError("type_ must be 'call' or 'put'")

    return option_price


def calcular_precios_de_opciones_para_K(S, k_values, T, r, volatilities, Npaths, type_='call'):
    prices_for_K = []

    for K_value in k_values:
        prices_for_K_value = [calculate_option_prices(S, K_value, T, r, volatilities[i, -1], type_) for i in range(Npaths)]
        prices_for_K.append(prices_for_K_value)

    return prices_for_K

def crear_dataframe_de_precios(S, T, r, volatilities, Npaths, k_values):
    call_prices = calcular_precios_de_opciones_para_K(S, k_values, T, r, volatilities, Npaths, type_='call')
    put_prices = calcular_precios_de_opciones_para_K(S, k_values, T, r, volatilities, Npaths, type_='put')

    data = {
        'K_value': k_values,
        'Call_Prices': call_prices,
        'Put_Prices': put_prices
    }

    df = pd.DataFrame(data)
    return df

def calcular_precios_de_opciones_para_K2(S, k_values, T, r, v_0, type_='call'):
    prices_for_K2 = []

    for K_value in k_values:
        prices_for_K_value = [calculate_option_prices(S, K_value, T, r, v_0, type_)]
        prices_for_K2.append(prices_for_K_value)

    return prices_for_K2

def crear_dataframe_de_precios2(S, T, r, v_0, k_values):
    call_prices2 = calcular_precios_de_opciones_para_K2(S, k_values, T, r, v_0, type_='call')
    put_prices2 = calcular_precios_de_opciones_para_K2(S, k_values, T, r, v_0, type_='put')

    data = {
        'K_value': k_values,
        'Call_Prices': call_prices2,
        'Put_Prices': put_prices2
    }

    df2 = pd.DataFrame(data)
    return df2

def calcular_precios_positivos_y_negativos(S, T, r, kappa, theta, v_0, rho, xi, steps, Npaths):
    prices_pos = generate_heston_paths(S, T, r, kappa, theta, v_0, rho, xi, steps, Npaths, return_vol=False)[:, -1]
    prices_neg = generate_heston_paths(S, T, r, kappa, theta, v_0, -rho, xi, steps, Npaths, return_vol=False)[:, -1]

    return prices_pos, prices_neg

def graficar_tail_density(prices_pos, prices_neg):
    fig, ax = plt.subplots()

    ax = sns.kdeplot(data=prices_pos, label="Positive", ax=ax)
    ax = sns.kdeplot(data=prices_neg, label="Negative", ax=ax)

    ax.set_title('Tail Density by Sign of Rho')
    plt.axis([0, 20000, 0, 0.00004])
    plt.xlabel('$S_T$')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

#Función para obtener la volatilidad implicita
def implied_vol(opt_value, S, K, T, r, type_='call'):
    def option_obj(sigma):
        if type_ == 'call':
            option_price = calculate_option_prices(S, K, T, r, sigma, type_='call')
        elif type_ == 'put':
            option_price = calculate_option_prices(S, K, T, r, sigma, type_='put')
        else:
            raise ValueError("type_ must be 'put' or 'call'")

        return abs(option_price - opt_value)

    # Ajusta los límites según tus necesidades
    bounds = (0.01, 6)

    res = minimize_scalar(option_obj, bounds=bounds, method='bounded')
    return res.x

def calcular_y_graficar_smileBSvolatility(S, k1, T, r, xi):
    # Valores de los precios de ejercicio
    strikes = k1

    # Listas para almacenar las volatilidades implícitas
    iv_values_c = []
    iv_values_p = []

    for K_value in strikes:
        C = calculate_option_prices(S, K_value, T, r, xi, type_='call')
        Pu = calculate_option_prices(S, K_value, T, r, xi, type_='put')
        iv_call = implied_vol(C, S, K_value, T, r)
        iv_values_c.append(iv_call)
        iv_put = implied_vol(Pu, S, K_value, T, r)
        iv_values_p.append(iv_put)

    # Grafica la volatilidad implícita para las opciones de compra
    plt.plot(strikes, iv_values_c)
    plt.ylabel('Implied Volatility')
    plt.xlabel('Strike')
    plt.axvline(S, color='black', linestyle='--', label='Spot Price')
    plt.title('Implied Volatility Call Smile from B&S Heston Model')
    plt.legend()
    plt.show()

    # Grafica la volatilidad implícita para las opciones de venta
    plt.plot(strikes, iv_values_p)
    plt.ylabel('Implied Volatility')
    plt.xlabel('Strike')
    plt.axvline(S, color='black', linestyle='--', label='Spot Price')
    plt.title('Implied Volatility Put Smile from B&S Heston Model')
    plt.legend()
    plt.show()


def calcular_y_graficar_volatilidades_implícitas(S, k1, T, r, prices_pos, prices_neg, rhoSV, xi, steps, Npaths):
    # Calcula las volatilidades implícitas para opciones de compra
    def calcular_volatilidades(tipo):
        opciones = []
        for L in k1:
            if tipo == 'call':
                P = np.mean(np.maximum(prices_pos - L, 0)) * np.exp(-r * T)
                opciones.append(P)
            elif tipo == 'put':
                P = np.mean(np.maximum(L - prices_neg, 0)) * np.exp(-r * T)
                opciones.append(P)

        ivs = [implied_vol(P, S, L, T, r, type_=tipo) for P, L in zip(opciones, k1)]

        plt.plot(k1, ivs)
        plt.ylabel('Implied Volatility')
        plt.xlabel('Strike')
        plt.axvline(S, color='black', linestyle='--', label='Spot Price')

        if tipo == 'call':
            plt.title('Implied Volatility Smile Call from Heston Model')
        elif tipo == 'put':
            plt.title('Implied Volatility Smile Put from Heston Model')

        plt.legend()
        plt.show()

    # Calcular y graficar volatilidades implícitas para opciones de compra
    calcular_volatilidades('call')

    # Calcular y graficar volatilidades implícitas para opciones de venta
    calcular_volatilidades('put')





