import os
import shutil
import sys

if len(sys.argv) <= 1:
  extensions = ['.bmp', '.gif', '.jpeg', '.jpg', '.png']
else:
  extension = sys.argv[1]

def filter_files(source_dir, destination_dir):
    count = 0
    if not os.path.exists(destination_dir):
      os.makedirs(destination_dir)
    for root, dirs, files in os.walk(source_dir):
      for file in files:
          os.rename(os.path.join(source_dir,file),f'{destination_dir}\\image{count}.jpg')
          count += 1

filter_files("Dog_Vs_Cat\\test_script_data\\cat_data", "Dog_Vs_Cat\\test_script_data\\clean_cat_data")
filter_files("Dog_Vs_Cat\\test_script_data\\dog_data", "Dog_Vs_Cat\\test_script_data\\clean_dog_data")