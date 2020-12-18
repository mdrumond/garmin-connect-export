import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

import os
from os import listdir
from os.path import isfile, join

import cProfile
import io
import pstats

import fitdecode

def profile(func):
  def wrapper(*args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    retval = func(*args, **kwargs)
    pr.disable()
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE  # 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    return retval

  return wrapper

def add_delta_col(df, orig_col, rolled_col, delta_col, is_time=False):
  df[rolled_col] = df[orig_col]
  df.iloc[-1, df.columns.get_loc(rolled_col)] = np.nan
  df[rolled_col] = np.roll(df[rolled_col], 1)
  df = df.fillna(method='bfill')
  if is_time:
    df[delta_col] = df.apply(lambda x: (x[orig_col] - x[rolled_col]).total_seconds(), axis=1)
  else:
    df[delta_col] = df.apply(lambda x: (x[orig_col] - x[rolled_col]), axis=1)
  return df

# @profile
def extract_points(fit_file_name):
  points = []
  with fitdecode.FitReader(fit_file_name) as fit:
    for frame in fit:
      if (isinstance(frame, fitdecode.FitDataMessage) and frame.name == 'sport' 
          and frame.get_field('sport').value != 'running'):
          return None


      if isinstance(frame, fitdecode.FitDataMessage) and frame.name == 'record':
        points.append([
          frame.get_field('timestamp').value, frame.get_field('distance').value,
          frame.get_field('heart_rate').value, frame.get_field('enhanced_speed').value,
          frame.get_field('enhanced_altitude').value])
  
  df = pd.DataFrame(points, columns=['time', 'distance', 'heart_rate', 'speed', 'altitude'])
    
  df = df.sort_values(by=['time'])
  df = df.reset_index()

  df = df.fillna(method='ffill')
  df = df.fillna(method='bfill')
  
  df['time'] = pd.to_datetime(df['time'], utc=True)
  df['time'] = df['time'].dt.tz_localize(tz=None)
  df = add_delta_col(df, 'time', 'time-start', 'time_delta', True)
  df = add_delta_col(df, 'distance', 'distance-start', 'distance_delta')
  df['distance_cumsum'] = df['distance_delta'].cumsum()
  df['time_cumsum'] = df['time_delta'].cumsum()

  return df

# @profile
def get_best_section(fit_file, df, section):
  # If the total distance of the workout is smaller then the section 
  # we're looking for we can skip this iteration.
  if df['distance_delta'].sum() < section:
    return None

  column_names = [
    'date', 'section', 'filename', 'time', 'distance', 'minutes_per_kilometer', 
    'total_distance', 'total_time']
  section_list = []
  date = df['time'].min()
  total_distance = df['distance_delta'].sum()
  total_time = df['time_delta'].sum()

  df_distance_cumsum = df['distance_cumsum']
  for i in range(len(df.index)):
    curr_row = df.loc[i]
    distance_cumsum = curr_row['distance_cumsum']
    time_cumsum = curr_row['time_cumsum']

    df_section = df[(df_distance_cumsum - distance_cumsum) >= section]
    if(len(df_section.index) != 0):
      time = df_section['time_cumsum'].iat[0] - time_cumsum
      distance_i = df_section['distance_cumsum'].iat[0] - distance_cumsum
      minutes_per_kilometer = (time/60)/(distance_i/1000)
      section_list.append([
        date, section, fit_file, time, distance_i, minutes_per_kilometer, 
        total_distance, total_time])

  df_output = pd.DataFrame(section_list, columns=column_names)
  return df_output.loc[df_output['minutes_per_kilometer'].idxmin()]


def process_file(input_folder, output_folder, fit_file, sections):
  df_final = pd.DataFrame(columns=['time', 'distance', 'minutes_per_kilometer'])
  path = os.path.join(os.path.abspath(''), input_folder, fit_file)
  df = extract_points(path)
  
  if df is None:
    return
  
  # Here we loop over sections
  for section in sections:
    s_best = get_best_section(fit_file, df, section)
    if s_best is None:
      return
    df_final = df_final.append(s_best)

  df_final.to_pickle(os.path.join(os.path.abspath(''), output_folder, fit_file + '.pkl'))

# All the sections you PB's for in meters:
def main():
  sections = [1000,(1000*1.60934),3000,(2000*1.60934),5000,10000,21097.5,30000,42195]
  input_folder = 'tracks'
  output_folder = 'output'
  path = os.path.join(os.path.abspath(''), input_folder)
  allfiles = [f for f in listdir(path) if isfile(join(path, f))]
  for fit_file in tqdm(allfiles):
    process_file(input_folder,output_folder,fit_file,sections)

if __name__ == "__main__":
    main()