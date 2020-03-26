import datetime
import os
import requests
import altair as alt
import pandas as pd
import numpy as np

def get_latest_data():
    base_url = "https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide-{}.xlsx"
    today = datetime.datetime.today()
    for day in range(5):
        date = today + datetime.timedelta(days=-day)
        url = base_url.format(datetime.datetime.strftime(date, '%Y-%m-%d'))
        response = requests.get(url, allow_redirects=True)
        if response.status_code == 200:
            return pd.read_excel(url)
        else:
            print(f"Data not available for date {date}")


def process_data_for_plot(df, min_deaths_number=80, n_death_date_origin=10):
    """
    Args:
        - min_deaths_number(int, default=80): only countries which have reached this threshold number of deaths are kept;
        - n_deaths_date_origin
    """
    # defines a date column
    df['date'] = pd.to_datetime(df['DateRep'], format='%Y-%m-%d')
    
    # remove useless columns
    df.drop(columns=['Day','Month','Year','GeoId','DateRep'], inplace=True)
    
    # convert deaths and cases
    for col in ['Cases', 'Deaths']:
        df[col] = df[col].astype(np.int)
        
    # rename columns
    df.rename(columns={
        'Cases':'cases',
        'Deaths':'deaths',
        'Countries and territories':'country',
    }, inplace=True)
    
    # sort the dataframe
    df.sort_values(by=['country','date'], ascending=True, inplace=True)
    
    # compute cumulative cases and cumulative_deaths
    df['cumul_deaths'] = df.groupby(['country'])['deaths'].cumsum()
    df['cumul_cases']  = df.groupby(['country'])['cases'].cumsum()

    # keep limited number of countries
    df_100_deaths = df.loc[ (df['cumul_deaths'] >= min_deaths_number) , : ].copy(deep=True)
    list_countries = df_100_deaths['country'].unique().tolist()
    df = df.loc[ df['country'].isin(list_countries) , : ]
    
    # compute the date of the 10th death
    df_date_10_deaths = df.loc[ (df['cumul_deaths'] >= n_death_date_origin) , : ].sort_values(by='date').drop_duplicates(subset='country', keep='first')
    df_date_10_deaths = df_date_10_deaths[['country','date']].sort_values(
        by='date', ascending=True
    ).rename(
        columns={'date':'date_10_deaths'}
    )

    # add this information to the main dataframe
    df = df.merge(right=df_date_10_deaths, left_on='country', right_on='country')
    df['days_since_10_death'] = ( (df['date'] - df['date_10_deaths']).dt.days ).astype(np.int)
    df = df.loc[ (df['days_since_10_death'] > 0) , : ]

    # sort the dataframe
    df.sort_values(by=['country','days_since_10_death'], inplace=True)

    return df
    
def get_altair_chart(df,x_max=30):
    line_deaths_cumul = alt.Chart(df).mark_line(point=True, clip=True).encode(
        x=alt.X(
            "days_since_10_death:Q",
            axis=alt.Axis(
                title="days since death number 10",  # Title of the x-axis.
                grid=False,  # Show the x-axis grid.
            ),
            scale = alt.Scale(domain=(0,x_max)),
        ),
        y=alt.Y(
            "cumul_deaths:Q",
            scale = alt.Scale(type='log'),
            axis = alt.Axis(
                title  = "death cumulative count",
                grid   = False,
                offset = 0,
            ),
        ),
        color = alt.Color(
            "country:N",
            scale=alt.Scale(scheme='category20'),
            legend=alt.Legend(orient='right'),
        ),
        tooltip = [
            alt.Tooltip("country"),
            alt.Tooltip("date"),
            alt.Tooltip("days_since_10_death"),
            alt.Tooltip("date_10_deaths"),
            alt.Tooltip("deaths"),
            alt.Tooltip("cumul_deaths"),
            alt.Tooltip("cases"),
            alt.Tooltip("cumul_cases"),
        ]
    )
    
    line_deaths_cumul_lin = alt.Chart(df).mark_line(point=True, clip=True).encode(
        x=alt.X(
            "days_since_10_death:Q",
            axis=alt.Axis(
                title="days since death number 10",  # Title of the x-axis.
                grid=False,  # Show the x-axis grid.
                #labelFontSize=18,
            ),
            scale = alt.Scale(domain=(0,x_max)),          
        ),
        y=alt.Y(
            "cumul_deaths:Q",
            axis = alt.Axis(
                title  = "death cumulative count",
                grid   = False,
                offset = 0,
            ),
        ),
        color = alt.Color(
            "country:N",
            scale=alt.Scale(scheme='category20'),
            legend=alt.Legend(orient='right'),
        ),
        tooltip = [
            alt.Tooltip("country"),
            alt.Tooltip("date"),
            alt.Tooltip("days_since_10_death"),
            alt.Tooltip("date_10_deaths"),
            alt.Tooltip("deaths"),
            alt.Tooltip("cumul_deaths"),
            alt.Tooltip("cases"),
            alt.Tooltip("cumul_cases"),
        ]
    )

    line_deaths_lin = alt.Chart(df).mark_line(point=True, clip=True).encode(
        x=alt.X(
            "days_since_10_death:Q",
            axis=alt.Axis(
                title="days since death number 10",  # Title of the x-axis.
                grid=False,  # Show the x-axis grid.
                #labelFontSize=18,
            ),
            scale = alt.Scale(domain=(0,x_max)),          
        ),
        y=alt.Y(
            "deaths:Q",
            axis = alt.Axis(
                title  = "death per day",
                grid   = False,
                offset = 0,
            ),
        ),
        color = alt.Color(
            "country:N",
            scale=alt.Scale(scheme='category20'),
            legend=alt.Legend(orient='right'),
        ),
        tooltip = [
            alt.Tooltip("country"),
            alt.Tooltip("date"),
            alt.Tooltip("days_since_10_death"),
            alt.Tooltip("date_10_deaths"),
            alt.Tooltip("deaths"),
            alt.Tooltip("cumul_deaths"),
            alt.Tooltip("cases"),
            alt.Tooltip("cumul_cases"),
        ]
    )
    

    # upper chart (logarthicmic y scale)
    chart_upper = alt.layer(
        line_deaths_cumul
    ).properties(
        height=400,
        width=700,
    )

    # lower chart
    chart_lower = alt.layer(
        line_deaths_cumul_lin
    ).properties(
        height=400,
        width=700,
    )

    chart_upper_2 = alt.layer(
        line_deaths_lin
    ).properties(
        height=400,
        width=700,
    )
    
    # concat the two plots vertically
    chart = alt.vconcat(
        chart_upper,
        chart_lower,
        chart_upper_2
    ).resolve_scale(
        color='independent'
    ).configure_axis(
        labelFontSize=16,
        titleFontSize=18
    ).configure_point(
        size=70
    ).configure_legend(
        titleFontSize=14,
        labelFontSize=14
    )

    return chart
