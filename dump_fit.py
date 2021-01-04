import fitdecode
import os
from tqdm import tqdm
from os.path import join

from utils import list_all_files, abs_path

import sys

def print_frame(frame):
  print(frame.name)
  for field in frame.fields:
    print("  ", field.name, field.value)

def main():
  with fitdecode.FitReader(sys.argv[1]) as fit:
    print("#### file:", sys.argv[1])
    for frame in fit:
      if isinstance(frame, fitdecode.FitDataMessage):
        print_frame(frame)

if __name__ == "__main__":
    main()