import pandas as pd
# import re

# CSV_PATH = "./MyData/RecGym.csv"

# # load in csv and clean any data that needs
# df = pd.read_csv(CSV_PATH)

# print(sorted(df["Subject"].unique()))

# # Only keep wrist data
# df = df[df["Position"] == "wrist"]

# df = df.drop(columns=["Position", "C_1"])
# df.columns = [
#     "participant",
#     "set",
#     "x_accel",
#     "y_accel",
#     "z_accel",
#     "x_gyro",
#     "y_gyro",
#     "z_gyro",
#     "exercise"
# ]

# print("Remaining rows:", len(df))
# print("Unique exercises:", df["exercise"].unique())
# print("Sessions:", sorted(df["set"].unique()))

# print(df.dtypes)

# # Print the result
# print(df['participant'].unique())

# print(df["set"].value_counts().sort_index())

# # subset = df.iloc[max(0, target_idx - 100) : target_idx + 101]
# def get_null_padding(df, column, label, idx_before, idx_after):
#     """
#     Get the 100 rows before and after the workout value to pad the dataset
#     """
#     wanted_indices = set()

#     label_indices = df[df[column] == label].index

#     for label in label_indices:
#         pos = df.index.get_loc(label)
        
#         # Determine the start and end of the slice
#         start_pos = max(0, pos - idx_before)
#         end_pos = min(len(df), pos + idx_after + 1) # +1 for inclusive slicing end in iloc
        
#         # Add the range of iloc positions to our set
#         for i in range(start_pos, end_pos):
#             # Convert iloc position back to original label index to handle non-range indices
#             wanted_indices.add(df.index[i])

#     # Select the rows using .loc and sort the index for correct order
#     # Convert the set of labels to a list for selection
#     wanted_indices_list = sorted(list(wanted_indices))
#     final_df = df.loc[wanted_indices_list]

#     return final_df

# df_armcurl = get_null_padding(df, 'exercise', 'ArmCurl', 100, 100)

# # print("Resulting DataFrame (1 row before, 2 rows after 'target'):")
# # print(df_armcurl.iloc[98:105])
# df_armcurl.to_csv("armcurl_sample.csv", index=False)
# index1 = df.indexOf
# subset = df.loc()
# df_armcurl = df[df["excercise"] == "ArmCurl"].copy()
# df_benchpress = df[df["excercise"] == "BenchPress"].copy()
# df_nulls = df[df["excercise"] == "Null"].copy()

# df_armcurl["excercise"] = df_armcurl["excercise"].str.replace(r"ArmCurl", r"1b", regex=True).str.lower()
# df_benchpress["excercise"] = df_benchpress["excercise"].str.replace(r"BenchPress", r"2a", regex=True).str.lower()
# df_nulls["excercise"] = df_nulls["excercise"].str.replace(r"Null", r"0", regex=True).str.lower()

# combined_df = pd.concat([df_nulls, df_armcurl, df_benchpress], ignore_index=True)

# # print(df_armcurl.sample(5))
# # print(df_benchpress.sample(5))
# # print(df_nulls.sample(5))

# # print(combined_df.sample(5))

# combined_df.to_csv("RecGym_augmented_v2.csv", index=False)



final_df = pd.read_csv("./MyData/ArmCurl_BenchPress_wrist_only.csv")
# print(sorted(out["participant"].unique()))
# print(sorted(out["set"].unique()))
# print(out[["parent", "child"]].drop_duplicates().sort_values(["parent", "child"]))
# print(out.groupby("ID").size().head())

print("IDs:", final_df["ID"].nunique())
print("Rows:", len(final_df))

# each ID should map to multiple rows
print(final_df.groupby("ID").size().head())

# ensure ID is unique per block
print(final_df.groupby("ID")[["participant", "set"]].nunique().head())