# We want parent classes --- 
# Dumbbell Arm Curl: Preacher Curl, Barbell Curls, Reverse Barbell Curls
# Bench Press: Incline barbel bench press, decline barbell bench press, flat dumbbell bench press,
#               incline dumbbell bench press, decline dumbbell bench press

import pandas as pd
import glob

input_csv = "data/*.csv"
output_file = "golden_dataset.csv"

all_files = glob.glob(input_csv)
if not all_files:
    raise FileNotFoundError(f"No files matched {input_csv}!")

df = pd.concat(map(pd.read_csv, all_files), ignore_index=True)

df.to_csv(output_file, index=False)

print(f"Successfully wrote {output_file} from {len(all_files)} files!")