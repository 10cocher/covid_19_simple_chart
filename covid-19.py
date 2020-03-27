import os
import libcovid19 as covid

# folder where to save output plots
out_dir = os.path.join(os.getcwd(), 'plots')
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# get the latest cases and death information
df = covid.get_latest_data()

# process input data
df = covid.process_data_for_plot(
    df,
    min_deaths_number = 80,
    n_death_date_origin = 10,
)

# prepare the plot with altair
chart = covid.get_altair_chart(df)

# save the chart as a html file
path_output_html_file = os.path.join(out_dir, 'charts_covid_19.html')
chart.save(path_output_html_file)
