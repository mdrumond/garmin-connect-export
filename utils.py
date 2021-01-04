import os
from os.path import isfile, join
from os import listdir

def log(message):
  with open('export.log', 'a+') as f:
    f.write(message)

def list_all_files(folder, condition = lambda x: True):
  path = os.path.join(os.path.abspath(''), folder)
  return [f for f in listdir(path) 
          if isfile(join(path, f)) and condition(f)]

def abs_path(*path):
  return join(os.path.abspath(''), *path)