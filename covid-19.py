import os

import libcovid19 as covid

# import altair as alt


# folder where to save output plots
out_dir = os.path.join(os.getcwd(), "plots")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# parameters
list_countries_to_keep = [
    "France",
    "Italy",
    "Spain",
    "Germany",
    "United_States_of_America",
    "South_Korea",
    "China",
    "United_Kingdom",
    "Netherlands",
    "Sweden",
    "Belgium",
]

# mode_threshold
# - "absolute" : use acutal death count
# - "relative" : use death count compare to population
mode_threshold = "relative"

# get the latest cases and death information
df = covid.get_latest_data()

# process input data
# df, captions = covid.process_data_for_plot(
#     df,
#     list_countries_to_keep=list_countries_to_keep,
#     mode_threshold="relative",
#     min_deaths_number=80,
#     min_deaths_percent=1.0,
#     n_death_date_origin=10,
#     n_percent_date_origin=1.0,
# )

# prepare the plot with altair
# chart = covid.get_altair_chart(df, captions, x_max=30)

# save the chart as a html file
# path_output_html_file = os.path.join(out_dir, "charts_covid_19.html")
#
# with alt.data_transformers.enable("default"):
#     chart.save(path_output_html_file)
