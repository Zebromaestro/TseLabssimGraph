import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import plotly
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import natsort
import glob
import os
from natsort import natsorted, os_sorted
from itertools import islice
import re

# from concurrent.futures import ThreadPoolExecutor
import mpire
from mpire import WorkerPool


def process_file(file_path):
    sort = pd.DataFrame()

    # Read the entire file_path once and split into parts
    with open(file_path, 'r') as f:
        lines = islice(f, 14)
        lines = list(lines)
    # contains list of string and numerical string
    flowrateStr, flowrate = re.split(r'  +', lines[12], maxsplit=1)
    flowrateStr = flowrateStr.strip()
    flowrate = float(flowrate)
    timeStr, time = re.split(r'  +', lines[13], maxsplit=1)
    timeStr = timeStr.strip()
    time = float(time)
    filename = os.path.basename(file_path)

    # Extract actual data (starting from row 14)
    df = pd.read_table(file_path, skiprows=14, sep=r'\s+')
    df = df.dropna(axis=1)

    # Create time and flowrate columns as new dataframes
    timeDf = pd.DataFrame({timeStr: np.full(len(df), time)})
    filenameDf = pd.DataFrame({'FileName': np.full(len(df), filename)})
    # flowrateDf = pd.DataFrame({flowrateStr: np.full(len(df), flowrate)})

    # Concatenate the time, flowrate, and the rest of the data
    dataDf = pd.concat([timeDf, df, filenameDf], axis=1)

    # Processing max values for sorting
    sort[timeStr] = [time]
    sort[flowrateStr] = [flowrate]
    sort['FileName'] = [filename]
    sort['Tmax(K)'] = df['Temperature(K)'].max()
    Tmax_i = df['Temperature(K)'].idxmax()
    sort['R_Tmax(cm)'] = df['Radius(cm)'][Tmax_i]

    # Check and add additional species maxima if present
    if 'CH' in df.columns:
        CHmax_i = df['CH'].idxmax()
        sort['R_CHmax(cm)'] = df['Radius(cm)'][CHmax_i]

    if 'CHA' in df.columns:
        CHAmax_i = df['CHA'].idxmax()
        sort['R_CHAmax(cm)'] = df['Radius(cm)'][CHAmax_i]

    if 'OH' in df.columns:
        OHmax_i = df['OH'].idxmax()
        sort['R_OHmax(cm)'] = df['Radius(cm)'][OHmax_i]

    if 'OHA' in df.columns:
        OHAmax_i = df['OHA'].idxmax()
        sort['R_OHAmax(cm)'] = df['Radius(cm)'][OHAmax_i]

    # create output for pickled df to store filename to flowrate,temp

    return dataDf, sort


def concat_chunk(*df_list_chunk):
    """Concatenate a chunk of DataFrames, handling multiple argument formats."""
    if len(df_list_chunk) == 1 and isinstance(df_list_chunk[0], list):
        df_list = df_list_chunk[0]
    else:
        df_list = list(df_list_chunk)
    return pd.concat(df_list, axis=0)


def parallel_concat(df_list, chunk_size=10000):
    """Concatenates a list of DataFrames in parallel using MPIRE."""
    chunks = [df_list[i:i + chunk_size] for i in range(0, len(df_list), chunk_size)]
    cpus = (int(os.environ['SLURM_CPUS_PER_TASK'])
            if os.environ.get('SLURM_CPUS_PER_TASK') else os.cpu_count())
    with WorkerPool(n_jobs=cpus) as pool:
        result_chunks = pool.map(concat_chunk, chunks)
    return pd.concat(result_chunks, axis=0).reset_index(drop=True)


def simParse(dir):
    filelist = os_sorted(glob.glob(dir + "/DATA*"))
    data_list = []
    sortmax_list = []

    print('executing multithreaded data processing')
    cpus = (int(os.environ['SLURM_CPUS_PER_TASK'])
            if os.environ.get('SLURM_CPUS_PER_TASK') else os.cpu_count())

    with WorkerPool(n_jobs=cpus, start_method='threading') as pool:
        results = pool.map(process_file, filelist)

    print('finished data processing. begin list append')
    for dataDf, sort in results:
        data_list.append(dataDf)
        sortmax_list.append(sort)

    print('concat into dataframes')
    data = parallel_concat(data_list)
    sortmax = parallel_concat(sortmax_list)

    print('finished dataframe processing')
    return data, sortmax


def calculate_area_under_curve(data_df, time_column='Time(s)'):
    radius_column = data_df.columns[1]
    grouped_data = pd.DataFrame()

    for t, group_df in data_df.groupby(time_column):
        areas = []
        element_names = []
        for i in range(5, len(data_df.columns) - 1):
            elements = group_df.iloc[:, i]
            radius = group_df[radius_column]
            area = np.trapz(elements, radius)
            areas.append(area)
            element_names.append(group_df.columns[i])

        header = f'Area Under Curve: {t}'
        area_data = pd.DataFrame({header: areas})
        if 'Species' in grouped_data:
            grouped_data = pd.concat([grouped_data, area_data], axis=1)
        else:
            grouped_data = pd.DataFrame({'Species': element_names})
            grouped_data = pd.concat([grouped_data, area_data], axis=1)
            grouped_data = grouped_data.head(len(element_names))

    return grouped_data


def graphScurve(file_path, df):
    fig = px.scatter(df, x='Mass flow rate(g/s)', y='Tmax(K)',
                     hover_data=['FileName'], template='plotly_dark')
    with open(f'{file_path}/a_scurve.html', 'w') as f:
        f.write(pio.to_html(fig, full_html=False, auto_play=False))


def simGraph(file_path, sm, df):
    fig1 = px.scatter(
        sm,
        x='Time(s)',
        y=[c for c in sm.columns if c not in ('FileName',)],
        hover_data=['FileName'],
        title='Sorted Maximums',
        template='plotly_dark'
    )

    colKeepy = ['Temperature(K)', 'C2H4', 'O2', 'N2', 'H2O',
                'CO2', 'CH', 'CHA', 'OH', 'OHA']
    value_vars = [c for c in colKeepy if c in df.columns]

    df_long = df.melt(
        id_vars=[c for c in ['Radius(cm)', 'Time(s)', 'FileName'] if c in df.columns],
        value_vars=value_vars,
        var_name='Species',
        value_name='Value'
    )

    fig2 = px.scatter(
        df_long,
        x='Radius(cm)',
        y='Value',
        color='Species',
        animation_frame='Time(s)',
        title='Simulation Data Across Time',
        template='plotly_dark'
    )

    with open(f'{file_path}/a_graph.html', 'w') as f:
        f.write(pio.to_html(fig1, full_html=False, auto_play=False))
        f.write(pio.to_html(fig2, full_html=False, auto_play=False))


def append2sortmax(sortmax_df, textfile):
    return textfile


if __name__ == "__main__":
    start = time.perf_counter()

    dir = "/Users/ryanmattana/Desktop/SPAM/sample data"
    print(f'Directory Input: {dir}')

    dataDf, sortmax = simParse(dir)

    dataDf.to_pickle(f'{dir}//dataDf.pkl')
    sortmax.to_pickle(f'{dir}//sortmax.pkl')

    print('Making S-Curve Graph')
    graphScurve(dir, sortmax)
    print('Making simGraph')
    simGraph(dir, sortmax, dataDf)

    end = time.perf_counter()
    print(f"Total execution time: {end - start:.3f} seconds")
