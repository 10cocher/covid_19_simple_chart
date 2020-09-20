import datetime
from typing import Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import requests


def get_latest_data() -> pd.DataFrame:
    base_url = "https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide-{}.xlsx"
    today = datetime.datetime.today()
    for day in range(5):
        date = today + datetime.timedelta(days=-day)
        url = base_url.format(datetime.datetime.strftime(date, "%Y-%m-%d"))
        response = requests.get(url, allow_redirects=True)
        if response.status_code == 200:
            return pd.read_excel(url)
        else:
            print(f"Data not available for date {date}")


def reorder_cols(res_df: pd.DataFrame, list_to_reorder: List[str]) -> pd.DataFrame:
    cols = list(res_df)

    to_add: List[str] = []
    for col in list_to_reorder:
        if col in cols:
            cols.remove(col)
            to_add.append(col)

    cols = to_add + cols

    res_df = res_df[cols]
    return res_df


def process_data_for_plot(
    df: pd.DataFrame,
    list_countries_to_keep: Optional[List[str]] = None,
    mode_threshold: str = "absolute",
    min_deaths_number: int = 80,
    min_deaths_percent: float = 1.0,
    n_death_date_origin: int = 10,
    n_percent_date_origin: float = 0.01,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Args:
        - df (pandas.DataFrame): input dataframe containing the covid-19 data;
        - list_countries_to_keep(list of str, None): list of countries to keep in the graph;
        - mode_threshold (str, default='absolute'): determine how
            - we choose the country qualified to be on the plot
            - the origin date for each country is determined
          mode_threshold should be one of the following
            - 'absolute':
                - use min_deaths_number and compares it to the absolute number of deaths;
                - use n_death_date_origin to determine an origin date for each ountry
            - 'relative':
                - use min_deaths_percent and compares it to the percentage (multiplied by one million) of population that has died
                - use n_death_date_origin to determine an origin date for each country
        - min_deaths_number(int, default=80): only countries which have reached this threshold number of deaths are kept;
        - min_deaths_percent(float, default=1.): only countries whose percentage of population dead from covid-19 (multiplied by one million) have reached this threshold are kept.
        - n_death_date_origin (int, default=10): the origin on the x-axis is determined as the date the total number of deaths reaches this threshold;
        - n_percent_date_origin (float, default=0.01): the origin on the x-axis is determined as the date the precent of hte population who died from covid-19 reaches this threshold.
    """
    # defines a date column
    df["date"] = pd.to_datetime(df["dateRep"], format="%Y-%m-%d")

    # remove useless columns
    df.drop(columns=["day", "month", "year", "dateRep"], inplace=True)

    # convert deaths and cases
    for col in ["cases", "deaths"]:
        df.rename(
            columns={
                "countriesAndTerritories": "country",
                "popData2019": "population",
            },
            inplace=True,
        )
        df.astype({"population": np.float32})

    # sort the dataframe
    df.sort_values(by=["country", "date"], ascending=True, inplace=True)

    # compute cumulative cases and cumulative_deaths
    df["cumul_deaths"] = df.groupby(["country"])["deaths"].cumsum()
    df["cumul_cases"] = df.groupby(["country"])["cases"].cumsum()

    # add normalized cases, deaths.... per population count
    for col in ["cases", "deaths", "cumul_cases", "cumul_deaths"]:
        colnew = col + "_per_habitant"
        df[colnew] = df[col].astype(np.float32) / df["population"] * np.float32(10 ** 6)

    df.drop(
        columns=[
            "cases",
            "cases_per_habitant",
            "cumul_cases",
            "cumul_cases_per_habitant",
        ],
        inplace=True,
    )

    # keep limited number of countries
    list_countries = get_list_of_countries_to_keep(
        df,
        mode_threshold,
        min_deaths_number,
        min_deaths_percent,
        list_countries_to_keep,
    )
    df = df.loc[df["country"].isin(list_countries), :]

    # compute the date where the graphs should start for each country
    df_date_threshold = get_initial_date_per_country(
        df, mode_threshold, n_death_date_origin, n_percent_date_origin,
    )

    # add this information to the main dataframe
    df = df.merge(
        right=df_date_threshold, how="left", left_on="country", right_on="country"
    )
    #
    df["days_since_date_threshold"] = (
        (df["date"] - df["date_threshold"]).dt.days
    ).astype(np.int)
    #
    df = df.loc[(df["days_since_date_threshold"] >= 0), :]

    # sort the dataframe
    df.sort_values(
        by=["country", "days_since_date_threshold"],
        ascending=[True, True],
        inplace=True,
    )

    # rearrange dataframe columns
    df.drop(columns=["geoId", "countryterritoryCode", "population"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = reorder_cols(
        df,
        [
            "country",
            "date_threshold",
            "date",
            "days_since_date_threshold",
            "deaths",
            "cumul_deaths",
            "cases",
            "cumul_cases",
            "deaths_per_habitant",
            "cumul_deaths_per_habitant",
            "cases_per_habitant",
            "cumul_cases_per_habitant",
        ],
    )

    captions = get_captions(mode_threshold, n_death_date_origin, n_percent_date_origin)

    return df, captions


def get_list_of_countries_to_keep(
    df: pd.DataFrame,
    mode_threshold: str,
    min_deaths_number: int,
    min_deaths_percent: float,
    list_countries_to_keep: Optional[List[str]] = None,
) -> List[str]:
    #
    if mode_threshold == "absolute":
        mask_threshold = df["cumul_deaths"] >= min_deaths_number
    elif mode_threshold == "relative":
        mask_threshold = df["cumul_deaths_per_habitant"] >= min_deaths_percent
    else:
        raise ValueError(
            "I do not know this value for mode_threshold: %s" % (mode_threshold)
        )
    #
    df_country_above_threshold = df.loc[mask_threshold, :].copy(deep=True)
    #
    list_countries = df_country_above_threshold["country"].unique().tolist()
    #
    if isinstance(list_countries_to_keep, list):
        list_countries = [
            country for country in list_countries if country in list_countries_to_keep
        ]
    #
    return list_countries


def get_initial_date_per_country(
    df: pd.DataFrame,
    mode_threshold: str,
    n_death_date_origin: int,
    n_percent_date_origin: float,
) -> pd.DataFrame:
    #
    if mode_threshold == "absolute":
        mask_date_threshold = df["cumul_deaths"] >= n_death_date_origin
    elif mode_threshold == "relative":
        mask_date_threshold = df["cumul_deaths_per_habitant"] >= n_percent_date_origin
    else:
        raise ValueError(
            "I do not know this value for mode_threshold: %s" % (mode_threshold)
        )
    #
    df_date_threshold = (
        df.loc[mask_date_threshold, :]
        .sort_values(by="date", ascending=True,)
        .drop_duplicates(subset="country", keep="first",)
    )
    #
    df_date_threshold = (
        df_date_threshold[["country", "date"]]
        .sort_values(by="date", ascending=True)
        .rename(columns={"date": "date_threshold"})
    )
    return df_date_threshold


def get_captions(
    mode_threshold: str, n_deaths_date_origin: int, n_percent_date_origin: float
) -> Dict[str, str]:
    if mode_threshold == "absolute":
        ylabel_cumul = "total death count"
        ylabel = "death per day"
        xlabel = "days since death number %i" % (n_deaths_date_origin)
        colname_deaths = "deaths"
        colname_cumul_deaths = "cumul_deaths"
    elif mode_threshold == "relative":
        ylabel_cumul = "total number of deaths in one million inhabitant"
        ylabel = "number of deaths in one million inhabitant per day"
        xlabel = (
            "days since the number of deaths is more than %i for one million inhabitants"
            % (n_percent_date_origin)
        )
        colname_deaths = "deaths_per_habitant"
        colname_cumul_deaths = "cumul_deaths_per_habitant"
    else:
        raise ValueError(
            "I do not know this value for mode_threshold: %s" % (mode_threshold)
        )
    #
    output_dict = {
        "xlabel": xlabel,
        "ylabel": ylabel,
        "ylabel_cumul": ylabel_cumul,
        "colname_deaths": colname_deaths,
        "colname_cumul_deaths": colname_cumul_deaths,
    }
    return output_dict


def get_altair_chart(
    df: pd.DataFrame, captions: Dict[str, str], x_max: float = 30
) -> alt.Chart:
    col_chart_1 = captions["colname_cumul_deaths"] + ":Q"
    col_chart_2 = captions["colname_deaths"] + ":Q"
    #
    point_dict = {"filled": True, "size": 20}
    #
    line_deaths_cumul = (
        alt.Chart(df)
        .mark_line(point=point_dict, clip=True)
        .encode(
            x=alt.X(
                "days_since_date_threshold:Q",
                axis=alt.Axis(
                    title=captions["xlabel"],  # Title of the x-axis.
                    grid=False,  # Show the x-axis grid.
                ),
                scale=alt.Scale(domain=(0, x_max)),
            ),
            y=alt.Y(
                col_chart_1,
                scale=alt.Scale(type="log"),
                axis=alt.Axis(title=captions["ylabel_cumul"], grid=False, offset=0,),
            ),
            color=alt.Color(
                "country:N",
                scale=alt.Scale(scheme="category20"),
                legend=alt.Legend(orient="right"),
            ),
            tooltip=[
                alt.Tooltip("country"),
                alt.Tooltip("date"),
                alt.Tooltip("days_since_date_threshold"),
                alt.Tooltip("date_threshold"),
                alt.Tooltip("deaths"),
                alt.Tooltip("cumul_deaths"),
                # alt.Tooltip("cases"),
                # alt.Tooltip("cumul_cases"),
            ],
        )
    )

    line_deaths_cumul_lin = (
        alt.Chart(df)
        .mark_line(point=point_dict, clip=True)
        .encode(
            x=alt.X(
                "days_since_date_threshold:Q",
                axis=alt.Axis(
                    title=captions["xlabel"],  # Title of the x-axis.
                    grid=False,  # Show the x-axis grid.
                    # labelFontSize=18,
                ),
                scale=alt.Scale(domain=(0, x_max)),
            ),
            y=alt.Y(
                col_chart_1,
                axis=alt.Axis(title=captions["ylabel_cumul"], grid=False, offset=0,),
            ),
            color=alt.Color(
                "country:N",
                scale=alt.Scale(scheme="category20"),
                legend=alt.Legend(orient="right"),
            ),
            tooltip=[
                alt.Tooltip("country"),
                alt.Tooltip("date"),
                alt.Tooltip("days_since_date_threshold"),
                alt.Tooltip("date_threshold"),
                alt.Tooltip("deaths"),
                alt.Tooltip("cumul_deaths"),
                # alt.Tooltip("cases"),
                # alt.Tooltip("cumul_cases"),
            ],
        )
    )

    line_deaths_lin = (
        alt.Chart(df)
        .mark_line(point=point_dict, clip=True)
        .encode(
            x=alt.X(
                "days_since_date_threshold:Q",
                axis=alt.Axis(
                    title=captions["xlabel"],  # Title of the x-axis.
                    grid=False,  # Show the x-axis grid.
                    # labelFontSize=18,
                ),
                scale=alt.Scale(domain=(0, x_max)),
            ),
            y=alt.Y(
                col_chart_2,
                axis=alt.Axis(title=captions["ylabel"], grid=False, offset=0,),
            ),
            color=alt.Color(
                "country:N",
                scale=alt.Scale(scheme="category20"),
                legend=alt.Legend(orient="right"),
            ),
            tooltip=[
                alt.Tooltip("country"),
                alt.Tooltip("date"),
                alt.Tooltip("days_since_date_threshold"),
                alt.Tooltip("date_threshold"),
                alt.Tooltip("deaths"),
                alt.Tooltip("cumul_deaths"),
                # alt.Tooltip("cases"),
                # alt.Tooltip("cumul_cases"),
            ],
        )
    )

    # upper chart (logarthicmic y scale)
    chart_upper = alt.layer(line_deaths_cumul).properties(height=400, width=700,)

    # lower chart
    chart_lower = alt.layer(line_deaths_cumul_lin).properties(height=400, width=700,)

    chart_upper_2 = alt.layer(line_deaths_lin).properties(height=400, width=700,)

    # concat the two plots vertically
    chart = (
        alt.vconcat(chart_upper, chart_lower, chart_upper_2)
        .resolve_scale(color="independent")
        .configure_axis(labelFontSize=16, titleFontSize=18)
        .configure_point(size=70)
        .configure_legend(titleFontSize=14, labelFontSize=14)
    )

    return chart
