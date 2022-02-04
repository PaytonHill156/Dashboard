import numpy as np
import pandas as pd

DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/"
    "master/csse_covid_19_data/csse_covid_19_time_series/"
    "time_series_covid19_{kind}_{group}.csv"
)

GROUPS = "world", "usa"
KINDS = "deaths", "cases"


# Exercise 1
def download_data(group, kind):
    """
    Reads in a single dataset from the John Hopkins GitHub repo
    as a DataFrame

    Parameters
    ----------
    group : "world" or "usa"

    kind : "deaths" or "cases"

    Returns
    -------
    DataFrame
    """
    group_change_dict = {"world": "global", "usa": "US"}
    kind_change_dict = {"deaths": "deaths", "cases": "confirmed"}
    group = group_change_dict[group]
    kind = kind_change_dict[kind]
    return pd.read_csv(DOWNLOAD_URL.format(kind=kind, group=group))


# Exercise 2
def read_all_data():
    """
    Read in all four CSVs as DataFrames

    Returns
    -------
    Dictionary of DataFrames
    """
    data = {}
    for group in GROUPS:
        for kind in KINDS:
            df = download_data(group, kind)
            data[f"{group}_{kind}"] = df
    return data


# Exercise 3
def write_data(data, directory, **kwargs):
    """
    Writes each raw data DataFrame to a file as a CSV

    Parameters
    ----------
    data : dictionary of DataFrames

    directory : string name of directory to save files i.e. "data/raw"

    kwargs : extra keyword arguments for the `to_csv` DataFrame method

    Returns
    -------
    None
    """
    for name, df in data.items():
        df.to_csv(f"{directory}/{name}.csv", **kwargs)


# Exercise 4
def read_local_data(group, kind, directory):
    """
    Read in one CSV as a DataFrame from the given directory

    Parameters
    ----------
    group : "world" or "usa"

    kind : "deaths" or "cases"

    directory : string name of directory to save files i.e. "data/raw"

    Returns
    -------
    DataFrame
    """
    return pd.read_csv(f"{directory}/{group}_{kind}.csv")


# Exercise 5
def run():
    """
    Run all cleaning and transformation steps

    Returns
    -------
    Dictionary of DataFrames
    """
    data = {}
    for group in GROUPS:
        for kind in KINDS:
            df = read_local_data(group, kind, "data/raw")
            data[f"{group}_{kind}"] = df
    return data


####### 04. Data Cleaning and Transformation notebook #######

# Exercise 6
def select_columns(df):
    """
    Selects the Country/Region column for world DataFrames and
    Province_State for USA

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    df : DataFrame
    """
    cols = df.columns

    # we don't need to know the group since the world and usa have
    # different column names for their areas
    areas = ["Country/Region", "Province_State"]
    is_area = cols.isin(areas)

    # date columns are the only ones with two slashes
    has_two_slashes = cols.str.count("/") == 2
    filt = is_area | has_two_slashes
    return df.loc[:, filt]


# Exercise 7
def run2():
    """
    Run all cleaning and transformation steps

    Returns
    -------
    Dictionary of DataFrames
    """
    data = {}
    for group in GROUPS:
        for kind in KINDS:
            df = read_local_data(group, kind, "data/raw")
            df = select_columns(df)
            data[f"{group}_{kind}"] = df
    return data


# Exercise 8
REPLACE_AREA = {
    "Korea, South": "South Korea",
    "Taiwan*": "Taiwan",
    "Burma": "Myanmar",
    "Holy See": "Vatican City",
    "Diamond Princess": "Cruise Ship",
    "Grand Princess": "Cruise Ship",
    "MS Zaandam": "Cruise Ship",
}


def update_areas(df):
    """
    Replace a few of the area names using the REPLACE_AREA dictionary.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    df : DataFrame
    """
    area_col = df.columns[0]
    df[area_col] = df[area_col].replace(REPLACE_AREA)
    return df


# Exercise 9
def run3():
    """
    Run all cleaning and transformation steps

    Returns
    -------
    Dictionary of DataFrames
    """
    data = {}
    for group in GROUPS:
        for kind in KINDS:
            df = read_local_data(group, kind, "data/raw")
            df = select_columns(df)
            df = update_areas(df)
            data[f"{group}_{kind}"] = df
    return data


# Exercise 10
def group_area(df):
    """
    Gets a single total for each area

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    df : DataFrame
    """
    grouping_col = df.columns[0]
    return df.groupby(grouping_col).sum()


# Exercise 11
def run4():
    """
    Run all cleaning and transformation steps

    Returns
    -------
    Dictionary of DataFrames
    """
    data = {}
    for group in GROUPS:
        for kind in KINDS:
            df = read_local_data(group, kind, "data/raw")
            df = select_columns(df)
            df = update_areas(df)
            df = group_area(df)
            data[f"{group}_{kind}"] = df
    return data


# Exercise 12
def transpose_to_ts(df):
    """
    Transposes the DataFrame and converts the index to datetime

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    df : DataFrame
    """
    df = df.T
    df.index = pd.to_datetime(df.index)
    return df


# Exercise 13
def run5():
    """
    Run all cleaning and transformation steps

    Returns
    -------
    Dictionary of DataFrames
    """
    data = {}
    for group in GROUPS:
        for kind in KINDS:
            df = read_local_data(group, kind, "data/raw")
            df = select_columns(df)
            df = update_areas(df)
            df = group_area(df)
            df = transpose_to_ts(df)
            data[f"{group}_{kind}"] = df
    return data


# Exercise 14
def fix_bad_data(df):
    """
    Replaces all days for each country where the value of
    deaths/cases is lower than the current maximum

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    DataFrame
    """
    mask = df < df.cummax()
    df = df.mask(mask).interpolate().round(0).astype("int64")
    return df


# Exercise 15
def run6():
    """
    Run all cleaning and transformation steps

    Returns
    -------
    Dictionary of DataFrames
    """
    data = {}
    for group in GROUPS:
        for kind in KINDS:
            df = read_local_data(group, kind, "data/raw")
            df = select_columns(df)
            df = update_areas(df)
            df = group_area(df)
            df = transpose_to_ts(df)
            df = fix_bad_data(df)
            data[f"{group}_{kind}"] = df
    return data


# Exercise 16
# Look at prepare.py file


from statsmodels.nonparametric.smoothers_lowess import lowess


# Exercise 17
def smooth(s, n):
    """
    Smooth data using lowess function from statsmodels

    Parameters
    ----------
    s : Series of daily deaths or cases

    n : int, number of points to be used by lowess function

    Returns
    -------
    Series of smoothed values with same index as the original
    """
    if s.values[0] == 0:
        # Filter the data if the first value is 0
        last_zero_date = s[s == 0].index[-1]
        s = s.loc[last_zero_date:]
        s_daily = s.diff().dropna()
    else:
        # If first value not 0, use it to fill in the
        # first missing value
        s_daily = s.diff().fillna(s.iloc[0])

    # Don't smooth data with less than 15 values
    if len(s_daily) < 15:
        return s

    y = s_daily.values
    frac = n / len(y)
    x = np.arange(len(y))
    y_pred = lowess(y, x, frac=frac, is_sorted=True, return_sorted=False)
    s_pred = pd.Series(y_pred, index=s_daily.index).clip(0)
    s_pred_cumulative = s_pred.cumsum()
    last_actual = s.values[-1]
    last_smoothed = s_pred_cumulative.values[-1]
    s_pred_cumulative *= last_actual / last_smoothed
    return s_pred_cumulative


from scipy.optimize import least_squares

# used in least_squares
def optimize_func(params, x, y, model):
    """
    Function to be passed as first argument to least_squares

    Parameters
    ----------
    params : sequence of parameter values for model

    x : x-values from data

    y : y-values from data

    model : function to be evaluated at x with params

    Returns
    -------
    Error between function and actual data
    """
    y_pred = model(x, *params)
    error = y - y_pred
    return error


# Exercise 18
def train_model(s, last_date, model, bounds, p0, **kwargs):
    """
    Train a model using scipy's least_squares function
    to find the fitted parameter values

    Parameters
    ----------
    s : pandas Series with data used for modeling

    last_date : string of last date to use for model

    model : function returning model values

    p0 : tuple of initial guess for parameters

    kwargs : extra keyword arguments forwarded to the least_squares function
                (ftol, xtol, max_nfev, verbose)

    Returns
    -------
    numpy array: fitted model parameters
    """
    y = s.loc[:last_date]
    n_train = len(y)
    x = np.arange(n_train)
    res = least_squares(optimize_func, p0, args=(x, y, model), bounds=bounds, **kwargs)
    return res.x


# Exercise 19
def get_daily_pred(model, params, n_train, n_pred):
    """
    Makes n_pred daily predictions given a trained model

    model : model that has already been trained

    params : parameters of trained model

    n_train : number of observations model trained on

    n_pred : number of predictions to make
    """
    x_pred = np.arange(n_train - 1, n_train + n_pred)
    y_pred = model(x_pred, *params)
    y_pred_daily = np.diff(y_pred)
    return y_pred_daily


# Exercise 20
def get_cumulative_pred(last_actual_value, y_pred_daily, last_date):
    """
    Returns the cumulative predicted values beginning with the
    first date after the last known date

    Parameters
    ----------
    last_actual_value : int, last recorded value

    y_pred_daily : array of predicted values

    last_date : string of last date used in model

    Returns
    -------
    Series with correct dates in the index
    """
    first_pred_date = pd.Timestamp(last_date) + pd.Timedelta("1D")
    n_pred = len(y_pred_daily)
    index = pd.date_range(start=first_pred_date, periods=n_pred)
    return pd.Series(y_pred_daily.cumsum(), index=index) + last_actual_value


# Exercise 21
def plot_prediction(s, s_pred, title=""):
    """
    Plots both the original and predicted values

    Parameters
    ----------
    s : Series of original data

    s_pred : Series of predictions

    title : title of plot

    Returns
    -------
    None
    """
    last_pred_date = s_pred.index[-1]
    ax = s[:last_pred_date].plot(label="Actual")
    s_pred.plot(label="Predicted")
    ax.legend()
    ax.set_title(title)


# Exercise 22
def predict_all(
    s, start_date, last_date, n_smooth, n_pred, model, bounds, p0, title="", **kwargs
):
    """
    Smooth, train, predict, and plot a Series of data

    Parameters
    ----------
    s : pandas Series with data used for modeling

    start_date : string of first date to use for model

    last_date : string of last date to use for model

    n_smooth : number of points of data to be used by lowess function

    n_pred : number of predictions to make

    model : function returning model values

    bounds : two-item tuple of lower and upper bounds of parameters

    p0 : tuple of initial guess for parameters

    title : title of plot

    kwargs : extra keyword arguments forwarded to the least_squares function
                (bounds, ftol, xtol, max_nfev, verbose)

    Returns
    -------
    Array of fitted parameters
    """
    # Smooth up to the last date
    s_smooth = smooth(s[:last_date], n=n_smooth)

    # Filter for the start of the modeling period
    s_smooth = s_smooth[start_date:]
    params = train_model(
        s_smooth, last_date=last_date, model=model, bounds=bounds, p0=p0, **kwargs
    )
    n_train = len(s_smooth)
    y_daily_pred = get_daily_pred(model, params, n_train, n_pred)
    last_actual_value = s.loc[last_date]
    s_cumulative_pred = get_cumulative_pred(last_actual_value, y_daily_pred, last_date)
    plot_prediction(s[start_date:], s_cumulative_pred, title=title)
    return params, y_daily_pred


# Exercise 23
def logistic_func(x, L, x0, k):
    """
    Computes the value of the logistic function

    Parameters
    ----------
    x : array of x values

    L : Upper asymptote

    x0 : horizontal shift

    k : growth rate
    """
    return L / (1 + np.exp(-k * (x - x0)))


# Exercise 24
def logistic_guess_plot(s, L, x0, k):
    """
    Plots the given series of data along with its
    estimated values using the logistic function.

    Parameters
    ----------
    s : Series of actual data

    L, x0, k : same as above

    Returns
    -------
    None
    """
    x = np.arange(len(s))
    y = logistic_func(x, L, x0, k)
    s_guess = pd.Series(y, index=s.index)
    s.plot()
    s_guess.plot()


# My guess
# logistic_guess_plot(italyc, 220_000, 50, .1)

# Exercise 25
def plot_ks(s, ks, L, x0):
    """
    Plots the data to be modeled along with the logistic curves
    for each of the given ks on the same plot. This function
    helps find good bounds for k in least_squares.

    Parameters
    ----------
    s : data to be modeled

    ks : list of floats

    L : Upper asymptote

    x0 : horizontal shift

    Returns
    -------
    None
    """
    start = s.index[0]
    index = pd.date_range(start, periods=2 * x0)
    x = np.arange(len(index))
    s.plot(label="smoothed", lw=3, title=f"L={L:,} $x_0$={x0}", zorder=3)
    for k in ks:
        y = logistic_func(x, L, x0, k)
        y = pd.Series(y, index=index)
        y.plot(label=f"k={k}").legend()


####### 09. Visualizations with Plotly #######

import plotly.graph_objects as go

# Exercise 26
def area_bar_plot(df, group, area, kind, last_date, first_pred_date):
    """
    Creates a bar plot of actual and predicted values for given kind 
    from one area
    
    Parameters
    ----------
    df - All data DataFrame
    
    group - "world" or "usa"
    
    area - A country or US state
    
    kind - "Daily Deaths", "Daily Cases", "Deaths", "Cases"

    last_date - last known date of data

    first_pred_date - first predicted date
    """
    df = df.query("group == @group and area == @area").set_index("date")
    df_actual = df[:last_date]
    df_pred = df[first_pred_date:]
    fig = go.Figure()
    fig.add_bar(x=df_actual.index, y=df_actual[kind], name="actual")
    fig.add_bar(x=df_pred.index, y=df_pred[kind], name="prediction")
    return fig
