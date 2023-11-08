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
        st.write("As you can see, here you go the historical data of the ticker and date selectioned")
        st.markdown(f"## Volatilities with the paths and steps selectioned")
        st.write("These charts show all the price and volatility paths you selected:")
        st.pyplot(fig)
        st.write("Each line represents a different trajectory, and we can observe how stock prices fluctuate in these trajectories as time progresses. "
                 "This is useful for comprehending the variability and uncertainty in stock prices according to the Heston model."
                 "In the case of volatility We can see that many of them behave quite similarly, and they also fluctuate as time progresses. "
                 "None of them go below 0, which is expected because we initialized our S to the last closing price, and for any asset, it wouldn't make sense for its volatility to go negative in the stock market.")
        st.markdown(f"## Volatilities with 10 paths and steps")
        st.write("These charts show only 10 paths to a better apreciation")
        st.pyplot(fig2)
        st.write("The blue line represents the mean of our Strike Prices series."
                 "These strike prices range from 30% below to 70% above this level to create a list of 20 possible values. "
                 "Beyond this blue line, prices that move upwards would provide an indication of profit because if it's set at your Strike Price and in the market, it reaches these levels, it would be beneficial for us."
                 " The same goes for selling; prices that fall below this Strike Price would generate a profit for us because we can sell our option at a higher price."
                 "We have 10 volatility trajectories, but this time the horizontal line in the graph represents the long-term volatility, which is represented by θ in the Heston model."
                 " It is simply the square root of θ and is used to display this value on the same scale as volatility on the Y-axis. The black line is a constant reference that indicates the level of long-term equilibrium volatility in the Heston model."
                 " In other words, it shows the value to which volatility tends to stabilize over time. "
                 )
        st.markdown(f"## Call and Put prices for all the K_values (Strike prices)")
        st.dataframe(df2)
        st.write("We can see all the K_values within the range of values we mentioned, along with all the possible calls and puts for the same Spot Price while iterating through the 5000 volatilities")
        st.markdown(f"### But, if you want to se only ONE price for each K_value? Here you go:")
        st.dataframe(df3)
        st.write("We can see how our table makes sense because it aligns with the reality and the correct functioning of options."
                 " In the case of call options, those starting with higher strike prices and lower spot prices are generally known as out-of-the-money options."
                 " This means that, at that moment, the strike price is above the current price of the underlying asset. "
                 "These options are typically cheaper because the market does not expect the underlying asset to reach that strike price before the expiration date."
                 " As the strike price increases, call options become more in-the-money, meaning the strike price is closer to or even below the current price of the underlying asset. "
                 "These options are more expensive due to the increasing probability that the asset will reach or exceed the strike price."
                 "On the other hand, put options follow the opposite logic. Those starting with lower strike prices and higher spot prices are usually out-of-the-money for put options."
                 " As the strike price increases, put options become more in-the-money and, therefore, more expensive.")
        st.markdown(f"## A little analysis if we have a positive and a negative rho")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig3)
        st.write("This graph shows how the probability density of final underlying asset prices varies when you change the correlation (ρ) in the Heston model. The shape of the distribution and the tails (extreme values) can change depending on whether the correlation is positive or negative."
                 " This is important in risk analysis and option valuation because correlation can affect the probability of extreme events in financial markets.")
        st.markdown(f"## Smile Volatility for Put and Call with B&S method")
        st.pyplot(fig8)
        st.pyplot(fig9)
        st.markdown(f"## Smile Volatility for Put and Call")
        st.pyplot(fig4)
        st.pyplot(fig5)
        st.write("Maybe you question why we have 2 graphs of the same topic (put and call) smile volatilities?."
                 "The first 2 graphs are Smile Volatility with the options prices with B&S method"
                 "The reason is because we want to show two different ways to obtain this"
                 "The reason for using the mean-max function to calculate the values of call and put options instead of directly using the price generated by the Black-Scholes function is to calculate the average value of options over multiple simulated price paths."
                 "In the context of option valuation in stochastic volatility models like the Heston model, multiple simulations of underlying asset prices are generated to represent market uncertainty. Each simulation results in a different option price based on the final prices of the simulations and the strike price."
                 "It calculates the intrinsic value of the call option (what you would gain if you exercised it) in each simulation."
                 "For sell choices (put options), we look at how much money you'd make if the thing you're selling becomes less valuable than the price you agreed to sell it at. We only count the cases where it's profitable."
                 "Then, we calculate the average value of these choices in today's money, taking into account factors like interest rates and how much time is left until you can make the choice."
                 "This process helps us figure out how much these options are worth when the future is uncertain. "
                 "It's particularly useful for complex models like the Heston model that deal with changing levels of uncertainty (volatility). Finally, we use these values to calculate implied volatilities, which are like the market's best guess about future uncertainty.")
        st.markdown(f"## Implied Volatility Surface Put")
        st.pyplot(fig6)
        st.markdown(f"## Implied Volatility Surface Call")
        st.write("The Implied Volatility Surface chart shows how implied volatility varies with the strike price (x-axis) and the tenor, which we chose as one year (time to expiration, y-axis). "
                 "Implied volatility is a measure of the market's expectation regarding the future volatility of the underlying asset.")
        st.pyplot(fig7)
        st.write("On the x-axis (Strike Price): It represents different options' strike prices. Each point along the x-axis corresponds to a specific strike price."
                 "On the y-axis (Tenor or Time to Expiration): It represents the time remaining until the options' expiration. Each point along the y-axis corresponds to a specific tenor value."
                 "On the z-axis (Implied Volatility): It represents implied volatility. The higher the value on the z-axis at a specific point on the surface, the higher the implied volatility associated with that strike price and tenor."
                 "Colors can indicate the magnitude of implied volatility, with darker colors possibly representing higher volatilities."
                 "These graph appears in that form because we are using B&S model."
                 )


