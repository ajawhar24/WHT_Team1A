import pandas as pd

CSV_PATH = "./MyData/DiffFreq/MyoGym_downsampled_40Hz.csv"
OUTPUT_PATH = "./MyData/DiffFreq/MyoGym_40Hz.csv"

NULL_LABEL = 99
NULL_PADDING = 100
MIN_EXERCISE_ROWS = 50

# Keep only these exercises
TARGET_CODES = {7, 13, 20}

# Final label mapping
LABEL_MAP = {
    99: {"parent": 0, "child": 0},
    7:  {"parent": 5, "child": 5},
    13: {"parent": 5, "child": 9},
    20: {"parent": 1, "child": 1},
}


def load_data(path):
    df = pd.read_csv(path, header=None)

    df.columns = [
        "x_accel",
        "y_accel",
        "z_accel",
        "x_gyro",
        "y_gyro",
        "z_gyro",
        "exercise",
        "participant",
    ]

    df["exercise"] = pd.to_numeric(df["exercise"], errors="coerce").astype(int)
    df["participant"] = pd.to_numeric(df["participant"], errors="coerce").astype(int)

    # remap participants to 10..19
    unique_ids = sorted(df["participant"].unique())
    # mapping = {old: new for old, new in zip(unique_ids, range(10, 10 + len(unique_ids)))}
    df["participant"] = df["participant"].astype(int)

    return df.reset_index(drop=True)


def find_target_blocks(df, target_codes):
    """
    Find contiguous target exercise blocks.
    A new block starts if:
      - exercise changes
      - participant changes
      - a non-target row appears
    """
    exercises = df["exercise"].to_numpy()
    participants = df["participant"].to_numpy()

    blocks = []
    in_block = False
    start = None
    current_exercise = None
    current_participant = None

    for i, ex in enumerate(exercises):
        part = participants[i]

        if ex in target_codes:
            if not in_block:
                in_block = True
                start = i
                current_exercise = ex
                current_participant = part
            elif ex != current_exercise or part != current_participant:
                blocks.append((start, i - 1, current_exercise, current_participant))
                start = i
                current_exercise = ex
                current_participant = part
        else:
            if in_block:
                blocks.append((start, i - 1, current_exercise, current_participant))
                in_block = False
                start = None
                current_exercise = None
                current_participant = None

    if in_block:
        blocks.append((start, len(df) - 1, current_exercise, current_participant))

    return blocks


def expand_with_existing_nulls(df, start, end, null_label=99, null_count=100):
    """
    Keep up to `null_count` contiguous existing null rows
    immediately before and after the target block.
    """
    values = df["exercise"].to_numpy()

    left = start
    taken = 0
    i = start - 1
    while i >= 0 and values[i] == null_label and taken < null_count:
        left = i
        taken += 1
        i -= 1

    right = end
    taken = 0
    i = end + 1
    while i < len(df) and values[i] == null_label and taken < null_count:
        right = i
        taken += 1
        i += 1

    return left, right


def build_output_block(df, left, start, end, right, exercise_code, block_id):
    raw_block = df.iloc[left:right + 1].copy().reset_index(drop=True)

    workout_start = start - left
    workout_end = end - left

    parent_vals = []
    child_vals = []

    for i in range(len(raw_block)):
        if workout_start <= i <= workout_end:
            code = LABEL_MAP[exercise_code]
        else:
            code = LABEL_MAP[NULL_LABEL]

        parent_vals.append(code["parent"])
        child_vals.append(code["child"])

    out = pd.DataFrame({
        "x_accel": raw_block["x_accel"],
        "y_accel": raw_block["y_accel"],
        "z_accel": raw_block["z_accel"],
        "x_gyro": raw_block["x_gyro"],
        "y_gyro": raw_block["y_gyro"],
        "z_gyro": raw_block["z_gyro"],
        "parent": parent_vals,
        "child": child_vals,
        "participant": raw_block["participant"].astype(int),
        "set": int(block_id),
        "ID": int(block_id),
    })

    return out


def preview_first_block(df, blocks):
    if not blocks:
        print("No blocks found.")
        return

    start, end, exercise_code, participant = blocks[0]
    left, right = expand_with_existing_nulls(df, start, end)

    raw_block = df.iloc[left:right + 1].copy().reset_index(drop=True)
    workout_start = start - left
    workout_end = end - left

    pre_null = workout_start
    exercise_rows = workout_end - workout_start + 1
    post_null = len(raw_block) - workout_end - 1

    print("\nFirst block preview")
    print(f"exercise={exercise_code}, participant={participant}")
    print(f"rows: total={len(raw_block)}, pre_null={pre_null}, exercise={exercise_rows}, post_null={post_null}")

    preview = raw_block[["exercise", "participant"]].copy()
    print("\nHead:")
    print(preview.head(8).to_string(index=False))

    print("\nAround exercise start:")
    print(preview.iloc[max(0, workout_start - 3): min(len(preview), workout_start + 5)].to_string(index=False))

    print("\nTail:")
    print(preview.tail(8).to_string(index=False))


def main():
    df = load_data(CSV_PATH)

    blocks = find_target_blocks(df, TARGET_CODES)

    # drop tiny fragments
    filtered_blocks = []
    dropped = 0

    for start, end, exercise_code, participant in blocks:
        block_len = end - start + 1
        if block_len >= MIN_EXERCISE_ROWS:
            filtered_blocks.append((start, end, exercise_code, participant))
        else:
            dropped += 1

    preview_first_block(df, filtered_blocks)

    output_blocks = []

    for block_id, (start, end, exercise_code, participant) in enumerate(filtered_blocks, start=1):
        left, right = expand_with_existing_nulls(
            df,
            start=start,
            end=end,
            null_label=NULL_LABEL,
            null_count=NULL_PADDING,
        )

        out_block = build_output_block(
            df=df,
            left=left,
            start=start,
            end=end,
            right=right,
            exercise_code=exercise_code,
            block_id=block_id,
        )

        output_blocks.append(out_block)

    final_df = pd.concat(output_blocks, ignore_index=True)
    final_df["dataset"] = "myogym"

    final_df = final_df[
        [
            "x_accel",
            "y_accel",
            "z_accel",
            "x_gyro",
            "y_gyro",
            "z_gyro",
            "parent",
            "child",
            "participant",
            "set",
            "ID",
            "dataset",
        ]
    ]

    final_df.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved:", OUTPUT_PATH)
    print("Dropped tiny fragments:", dropped)
    print("Shape:", final_df.shape)
    print("Unique participants:", sorted(final_df["participant"].unique()))
    print("Unique sets:", sorted(final_df["set"].unique())[:10])
    print("Unique labels:")
    print(final_df[["parent", "child"]].drop_duplicates().sort_values(["parent", "child"]))
    print("Number of blocks:", final_df["ID"].nunique())


if __name__ == "__main__":
    main()