import sys
from typing import Any, Dict, List

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output

import libcovid19 as covid

print(sys.version)


print(html.__version__)

# parameters by default
# ---------------------
mode_default: str = "cases"
list_countries_to_keep_default: List[str] = ["France", "Germany"]
per_habitant_default: bool = True

# fetch data
# ----------
df = covid.get_latest_data()
df = covid.pre_process_data(df)
list_available_countries: List[str] = df["countriesAndTerritories"].unique().tolist()

# list available countries
# ------------------------
list_countries_reduced: List[str] = [
    "France",
    "Germany",
    "Belgium",
    "United_Kingdom",
    "Italy",
    "Spain",
    "Austria",
    "Portugal",
    "Switzerland",
    "Iceland",
    "South_Korea",
    "Sweden",
    "Denmark",
    "Norway",
]

list_countries_proposed: List[str] = []
for country in sorted(list_countries_reduced):
    if country not in list_available_countries:
        print(f"Warning: {country} not available...")
    else:
        list_countries_proposed.append(country)

# list of options for the countries check-list
# --------------------------------------------
options_countries_to_be_plotted: List[Dict[str, str]] = []
for country in list_countries_proposed:
    options_countries_to_be_plotted.append({"label": country, "value": country})

# data for selector cases/deaths
# ------------------------------
options_cases_deaths: List[Dict[str, str]] = [
    {"label": "deaths", "value": "deaths"},
    {"label": "cases", "value": "cases"},
]

# data for "per-habitant" RadioItems
# ----------------------------------
options_per_habitant: List[Dict[str, Any]] = [
    {"label": "numbers per inhabitant", "value": "per_habitant"},
    {"label": "absolute numbers", "value": "absolute"},
]


# initializes the dash app
# -----------------------
app = dash.Dash(external_stylesheets=[dbc.themes.MINTY])


# Tile to choose whether data is normalized or not
# ------------------------------------------------
card_normalization = dbc.Card(
    [
        dbc.CardHeader("Normalization"),
        dbc.CardBody(
            dcc.RadioItems(
                id="radio-per-habitant",
                options=options_per_habitant,
                value="per_habitant",
            )
        ),
    ]
)

# Tile to choose quantity to be plotted
# -------------------------------------
card_cases_deaths = dbc.Card(
    [
        dbc.CardHeader("Quantity"),
        dbc.CardBody(
            dcc.Dropdown(
                id="dropdown-casesdeaths", options=options_cases_deaths, value="cases"
            )
        ),
    ]
)

# Tile to add countries
# ---------------------
card_countries = dbc.Card(
    [
        dbc.CardHeader("Add countries..."),
        dbc.CardBody(
            dcc.Checklist(
                id="country-checklist",
                options=options_countries_to_be_plotted,
                value=["France", "Germany"],
                labelStyle={"display": "block"},
            )
        ),
    ]
)

# first column
# ------------
first_col = [
    card_normalization,
    html.Br(),
    card_cases_deaths,
    html.Br(),
    card_countries,
]

# second column
# -------------
second_col = [dcc.Graph(id="graph-covid")]

app.layout = html.Div(
    children=[
        dbc.Row(
            [
                # first column
                # ------------
                dbc.Col(html.Div(first_col), width={"size": 2}),
                # second column
                # -------------
                dbc.Col(html.Div(second_col), width={"size": 9}),
            ]
        )
    ]
)


@app.callback(
    Output("graph-covid", "figure"),
    [
        Input("radio-per-habitant", "value"),
        Input("dropdown-casesdeaths", "value"),
        Input("country-checklist", "value"),
    ],
)
def print_dropdown_results(
    per_habitant: str, cases_deaths: str, country_checklist: List[str]
) -> go.Figure:
    # convert string argument `per_habitant` into a boolean
    bool_per_habitant = per_habitant == "per_habitant"

    # call the function to create the plot
    fig = covid.get_plot(
        df,
        mode=cases_deaths,
        list_countries_to_keep=country_checklist,
        per_habitant=bool_per_habitant,
    )
    return fig


if __name__ == "__main__":
    app.run_server(port=8007, debug=True)
