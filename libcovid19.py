import datetime
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

COLNAME_DATE = "dateRep"
COLNAME_COUNTRY = "countriesAndTerritories"
COLNAME_DEATHS = "deaths"
COLNAME_CASES = "cases"
COLNAME_POP = "popData2019"


def get_latest_data() -> pd.DataFrame:
    base_url = "https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide-{}.xlsx"
    today = datetime.datetime.today()
    for day in range(5):
        date = today + datetime.timedelta(days=-day)
        url = base_url.format(datetime.datetime.strftime(date, "%Y-%m-%d"))
        response = requests.get(url, allow_redirects=True)
        if response.status_code == 200:
            return pd.read_excel(url, dtype=str)
        else:
            print(f"Data not available for date {date}")


def pre_process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Adjust columns types."""
    list_cols_datetime: List[str] = [COLNAME_DATE]
    list_cols_floats: List[str] = [COLNAME_DEATHS, COLNAME_CASES, COLNAME_POP]
    #
    for col in list_cols_datetime:
        df[col] = pd.to_datetime(df[col])
    for col in list_cols_floats:
        df[col] = df[col].astype(np.float32)
    #
    return df


def get_plot(
    df_input: pd.DataFrame,
    mode: str,
    list_countries_to_keep: List[str],
    per_habitant: bool = False,
) -> go.Figure:
    """Plot the time series of cases/deaths for several countries with plotly.

    Parameters
    ----------
    df_input: pd.DataFrame
        Input data

    mode: {"deaths", "cases"}
        Quantity that should be plotted

    list_countries_to_keep: list of str
        List of countries to be represented in the plot

    per_habitant: bool
        If True, normalize the time series of deaths or cases by the number of inhabitant of each country.

    Returns
    -------
    fig: go.Figure
        Plot of the time series with plotly.
    """
    # make a copy to preserve original data
    df = df_input.copy(deep=True)

    # select the column to be plotted
    if mode == "deaths":
        colname_y = COLNAME_DEATHS
    elif mode == "cases":
        colname_y = COLNAME_CASES
    else:
        raise ValueError(f"unknown value for mode: {mode}")

    # normalize data if necessary
    if per_habitant:
        df[colname_y] = df[colname_y] / df[COLNAME_POP] * np.float32(10 ** 6)

    # determine axis labels
    if per_habitant:
        title = f"number of {mode} per day for 1 million habitant"
        hovertemplate = "%{y:.2f}"
    else:
        title = f"number of {mode} per day"
        hovertemplate = "%{y:6}"

    # initialize the figure
    fig = go.Figure()
    maxval: float = 0.0

    for country in list_countries_to_keep:
        #
        df_country = df.query(expr=f"{COLNAME_COUNTRY} == @country")
        maxval = max(maxval, df_country[colname_y].max())
        #
        fig.add_trace(
            go.Scattergl(
                x=df_country[COLNAME_DATE],
                y=df_country[colname_y],
                name=country,
                mode="markers+lines",
                hovertemplate=hovertemplate + f"<extra>{country}</extra>",
            )
        )

    fig.update_yaxes(range=[0, maxval])
    fig.update_layout(title={"text": title}, hovermode="x")

    return fig
