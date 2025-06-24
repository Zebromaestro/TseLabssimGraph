import numpy as np
import matplotlib.pyplot as plt
import polars as pl
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

import mpire
from mpire import WorkerPool


def process_file(file_path):
    # Read header information
    with open(file_path, 'r') as f:
        lines = list(islice(f, 14))

    # Extract flow rate and time values
    flowrate_val = float(lines[12].split()[-1])
    time_val = float(lines[13].split()[-1])
    filename = os.path.basename(file_path)

    # Read data with pandas (flexible whitespace handling)
    df_pd = pd.read_csv(
        file_path,
        skiprows=14,
        sep=r'\s+',
        engine='python',
        header=None,
        skip_blank_lines=False
    )

    # Find the first non-empty row for headers
    header_row = None
    for i in range(len(df_pd)):
        if not df_pd.iloc[i].isnull().all():
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"No valid header found in {file_path}")

    # Set headers and remove header row
    df_pd.columns = df_pd.iloc[header_row].values
    df_pd = df_pd.iloc[header_row + 1:]

    # Remove columns with all null values
    df_pd = df_pd.dropna(axis=1, how='all')

    # Convert to polars
    df = pl.from_pandas(df_pd)

    # Convert columns to float except FileName
    for col in df.columns:
        if col != 'FileName':
            df = df.with_columns(pl.col(col).cast(pl.Float64))

    # Add metadata
    data_df = df.with_columns(
        pl.lit(time_val).alias('Time(s)'),
        pl.lit(filename).alias('FileName')
    )

    # Create sortmax row
    sortmax_row = {
        'Time(s)': time_val,
        'Mass flow rate(g/s)': flowrate_val,
        'FileName': filename
    }

    # Calculate maxima
    if 'Temperature(K)' in df.columns:
        temp_idx = df['Temperature(K)'].arg_max()
        sortmax_row['Tmax(K)'] = df['Temperature(K)'].max()
        sortmax_row['R_Tmax(cm)'] = df['Radius(cm)'][temp_idx]

    for species in ['CH', 'CHA', 'OH', 'OHA']:
        if species in df.columns:
            species_idx = df[species].arg_max()
            sortmax_row[f'R_{species}max(cm)'] = df['Radius(cm)'][species_idx]

    return data_df, pl.DataFrame(sortmax_row)


def concat_chunk_polars(*dfs):
    """Concatenate any number of DataFrames"""
    return pl.concat(dfs)


def parallel_concat_polars(df_list):
    if not df_list:
        return pl.DataFrame()

    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))

    # Process in parallel: each worker gets a portion of the list
    with WorkerPool(n_jobs=cpus) as pool:
        # Single job that concatenates everything
        if len(df_list) == 1:
            return df_list[0]
        # Split list into chunks for parallel processing
        chunks = np.array_split(df_list, cpus)
        results = pool.map(concat_chunk_polars, chunks)

    return pl.concat(results)


def simParse(dir):
    filelist = os_sorted(glob.glob(os.path.join(dir, "DATA*")))
    data_list = []
    sortmax_list = []

    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))

    with WorkerPool(n_jobs=cpus, start_method='threading') as pool:
        results = pool.map(process_file, filelist, progress_bar=True)

    for data_df, sortmax_df in results:
        data_list.append(data_df)
        sortmax_list.append(sortmax_df)

    data = parallel_concat_polars(data_list)
    sortmax = parallel_concat_polars(sortmax_list)

    return data, sortmax


def graphScurve(file_path, df):
    df_pd = df.to_pandas()
    fig = px.scatter(
        df_pd,
        x='Mass flow rate(g/s)',
        y='Tmax(K)',
        hover_data=['FileName'],
        template='plotly_dark'
    )
    pio.write_html(fig, os.path.join(file_path, 'a_scurve.html'), auto_play=False)


def simGraph(file_path, sm, df):
    sm_pd = sm.to_pandas()
    df_pd = df.to_pandas()

    # S-curve plot
    fig1 = px.scatter(
        sm_pd,
        x='Time(s)',
        y=[c for c in sm_pd.columns if c not in ('FileName',)],
        hover_data=['FileName'],
        title='Sorted Maximums',
        template='plotly_dark'
    )

    # Animated species plot
    col_keep = ['Temperature(K)', 'C2H4', 'O2', 'N2', 'H2O',
                'CO2', 'CH', 'CHA', 'OH', 'OHA']
    value_vars = [c for c in col_keep if c in df_pd.columns]

    # Ensure Time(s) is numeric for animation
    df_pd['Time(s)'] = pd.to_numeric(df_pd['Time(s)'])

    df_long = df_pd.melt(
        id_vars=['Radius(cm)', 'Time(s)', 'FileName'],
        value_vars=value_vars,
        var_name='Species',
        value_name='Value'
    )

    fig2 = px.line(  # Better for continuous data
        df_long,
        x='Radius(cm)',
        y='Value',
        color='Species',
        animation_frame='Time(s)',
        title='Species Concentration Over Time',
        template='plotly_dark'
    )

    # Combine plots
    with open(os.path.join(file_path, 'a_graph.html'), 'w') as f:
        f.write(pio.to_html(fig1, full_html=False, include_plotlyjs='cdn', auto_play=False))
        f.write(pio.to_html(fig2, full_html=False, include_plotlyjs='cdn', auto_play=False))


if __name__ == "__main__":
    dir_path = "/Users/ryanmattana/Desktop/SPAM/sample data"
    print(f'Directory Input: {dir_path}')

    data_df, sortmax = simParse(dir_path)

    # Write outputs
    data_df.write_parquet(os.path.join(dir_path, 'dataDf.parquet'))
    sortmax.write_parquet(os.path.join(dir_path, 'sortmax.parquet'))

    print('Making S-Curve Graph')
    graphScurve(dir_path, sortmax)
    print('Making Simulation Graphs')
    simGraph(dir_path, sortmax, data_df)