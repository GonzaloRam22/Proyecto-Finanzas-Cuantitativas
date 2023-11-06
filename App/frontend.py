import textwrap
import datetime as dt

from functions import *
import streamlit as st
#import altair as alt



st.markdown(f"# Heston Model Application")

with (st.form("exchange-form")):
    st.write("Heston Model")
    ticker = st.text_input(label="Ticker:", value="AMZN")
    T = st.number_input(label = "Time: ")
    r = st.number_input(label = "Interest Rate: ")
    steps = st.slider("Number of steps", 1,252,1)

    Npaths = st.slider("Number of paths to simulate ", 300,10000,300)



    initial_date = st.date_input(
        label="Select initial date:",
        value=(dt.datetime.today() - dt.timedelta(days=1)),
        max_value=dt.datetime.today(),
    ).strftime("%Y-%m-%d")

    end_date = st.date_input(
        label="Select Final date:",
        value=(dt.datetime.today() - dt.timedelta(days=1)),
        max_value=dt.datetime.today(),
    ).strftime("%Y-%m-%d")

    submitted = st.form_submit_button("Submit")
    if submitted:


        # Obtener datos

        historical_data, S, v_0,theta, kappa, xi, rho = generate_data(ticker, initial_date, end_date)

        prices, volatilities = generate_heston_paths(S, T, r, kappa, theta, v_0, rho, xi, steps, Npaths, return_vol=True)

        # Graficar

        fig = graficar_simulacion_precios_y_volatilidades(prices, volatilities, steps)
        y = 130
        fig2 = graficar_simulacion_precio_y_volatilidad2(prices, volatilities, steps, y, theta)
        k1 = np.linspace(S * 0.3, S * 1.7, num=20)
        df2 = crear_dataframe_de_precios(S, T, r, volatilities, Npaths, k1)
        df3 = crear_dataframe_de_precios2(S, T, r, v_0, k1)

        prices_pos, prices_neg = calcular_precios_positivos_y_negativos(S, T, r, kappa, theta, v_0, rho, xi,
                                                                        steps, Npaths)
        fig3 = graficar_tail_density(prices_pos, prices_neg)

        prices_pos2, prices_neg2 = calcular_precios_positivos_y_negativos(S, T, r, kappa, theta, v_0, rho, xi,
                                                                          steps, Npaths)
        fig4 = graficar_smile_put(S, k1, prices_neg2, T, r)
        fig5 = graficar_smile_call(S, k1, prices_pos2, T, r)

        tipo = "put"
        tenors = np.linspace(0.1, 1, num=20)
        ivs = calcular_volatilidades(tipo, k1, prices_pos, prices_neg, r, T, S)
        fig6 = plot_implied_volatility_surface(k1, tenors, ivs)

        tipo2 = "call"
        tenors = np.linspace(0.1, 1, num=20)
        ivscall = calcular_volatilidades(tipo2, k1, prices_pos, prices_neg, r, T, S)
        fig7 = plot_implied_volatility_surface(k1, tenors, ivscall)

        fig8 = calcular_y_graficar_smileBSvolatility_calls(S, k1, T, r, xi)
        fig9 = calcular_y_graficar_smileBSvolatility_puts(S, k1, T, r, xi)


        #grafica pyplot
        #dataframe igual
        #texto write

        st.markdown(f"## Here you can see your selectioned data")
        st.dataframe(historical_data)

        st.markdown(f"## Volatilities with the paths and steps selectioned")
        st.pyplot(fig)
        st.markdown(f"## Volatilities with 10 paths and steps")
        st.pyplot(fig2)
        st.markdown(f"## Call and Put prices for all the K_values (Strike prices)")
        st.dataframe(df2)
        st.markdown(f"### But, if you want to se only ONE price for each K_value? Here you go:")
        st.dataframe(df3)
        st.markdown(f"## A little analysis if we have a positive and a negative rho")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig3)
        st.markdown(f"## Smile Volatility for Put and Call with B&S method")
        st.pyplot(fig8)
        st.pyplot(fig9)
        st.markdown(f"## Smile Volatility for Put and Call")
        st.pyplot(fig4)
        st.pyplot(fig5)
        st.markdown(f"## Implied Volatility Surface Put")
        st.pyplot(fig6)
        st.markdown(f"## Implied Volatility Surface Call")
        st.pyplot(fig7)
        st.write("Las gráficas...")


