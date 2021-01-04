import pandas as pd
import numpy as np
import os
from os.path import isfile, join

import math

from utils import log, list_all_files, abs_path

def read_dataframes(folder):
  pickle_files = list_all_files(folder)
  
  dfs = [pd.read_pickle(abs_path(folder,pickle_file))
         for pickle_file in pickle_files]
  return pd.concat(dfs)

def record_history(dfs, section):
  records = []
  curr_record_date = np.datetime64('now')
  for row in dfs.loc[(dfs['section']==section)].sort_values(by='time').itertuples():
    if row.date < curr_record_date:
      curr_record_date = row.date
      records.append([row.date, row.time, row.minutes_per_kilometer])
  return pd.DataFrame(records, columns=[
    'date', '%0.0f_time' % section, '%0.0f_minutes_per_kilometer' % section])

def main():
  input_folder = 'output'
  output_folder = 'csvs'
  dfs = read_dataframes(input_folder)
  sections = [1000,(1000*1.60934),3000,(2000*1.60934),5000,10000,21097.5,30000,42195]

  df = pd.concat([record_history(dfs, section) for section in sections])
  
  df.to_csv(abs_path(output_folder,"records.csv"))
  #for rh,s in zip(record_histories, sections):
  #  rh.to_csv(abs_path(output_folder,"%0.0f.csv" % s))
if __name__ == "__main__":
    main()