from glob import glob
import os

for root, dirs, files in os.walk(r"C:\project\dataset_src\nodig_data\09.05.19"):
    for file in files:
        if file.endswith('.avi'):
            print(os.path.join(root, file))

for file in glob(r"C:\project\dataset_src\nodig_data\09.05.19\*\*\*.avi"):
    print(file)
