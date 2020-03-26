import libcovid19 as covid

df = covid.get_latest_data()
df = covid.process_data_for_plot(df, min_deaths_number=80, n_death_date_origin=10)
chart = covid.get_altair_chart(df)

chart.save('charts_covid_19.html')
