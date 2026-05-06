"""
remove_gravity_quaternion.py

Removes gravity from myogym IMU data using a Madgwick sensor fusion filter
(quaternion-based). The other datasets (ourdata, lean) are passed through
untouched.

Method:
  For each participant, the Madgwick filter (via the imufusion library) fuses
  the accelerometer and gyroscope readings at each timestep to estimate the
  sensor's current 3D orientation as a quaternion. This orientation is used to
  rotate the raw accelerometer vector into a fixed world frame, after which
  gravity (always pointing in a fixed world-frame direction) is subtracted
  automatically. The result is the linear (motion-only) acceleration in the
  world frame.

  This is superior to static subtraction when the sensor rotates during the
  exercise, because the gravity vector is removed dynamically at every sample
  rather than using a single fixed estimate.

Comparison output:
  The script also runs the static subtraction method and prints a side-by-side
  comparison of residual statistics for every participant.

Dependencies:
  pip install imufusion pandas numpy

Usage:
  python remove_gravity_quaternion.py \
      --input  golden_dataset_v8_trimmed.csv \
      --output golden_dataset_gravity_removed_quat.csv \
      [--gain 0.5] \
      [--gyro_range 200] \
      [--gyro_thresh 15]
"""

import argparse
import numpy as np
import pandas as pd

try:
    import imufusion
except ImportError:
    raise ImportError("Please install imufusion:  pip install imufusion")


ACCEL_COLS = ["x_accel", "y_accel", "z_accel"]
GYRO_COLS  = ["x_gyro",  "y_gyro",  "z_gyro"]
SAMPLE_RATE = 20.0


# ---------------------------------------------------------------------------
# Gravity removal methods
# ---------------------------------------------------------------------------

def remove_gravity_static(accel: np.ndarray, gyro: np.ndarray,
                           gyro_thresh: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate a single fixed gravity vector from near-static samples, then
    subtract it from all samples.

    Returns (corrected_accel, gravity_vector).
    """
    gyro_mag   = np.sqrt((gyro ** 2).sum(axis=1))
    static_mask = gyro_mag < gyro_thresh

    if static_mask.sum() < 5:
        raise ValueError(
            f"Only {static_mask.sum()} static samples found at gyro_thresh={gyro_thresh}. "
            "Try increasing --gyro_thresh."
        )

    grav_vec = accel[static_mask].mean(axis=0)
    return accel - grav_vec, grav_vec


def remove_gravity_quaternion(accel: np.ndarray, gyro: np.ndarray,
                               gain: float, gyro_range: int) -> np.ndarray:
    """
    Use the Madgwick filter (imufusion) to fuse accel + gyro into a quaternion
    orientation estimate at every timestep. imufusion.Ahrs.earth_acceleration
    returns the accelerometer reading rotated into the world frame with gravity
    already removed.

    gain          : Madgwick filter gain (0–1). Higher = trusts accel more,
                    lower = trusts gyro integration more. 0.5 is a reasonable
                    default for bench press at 20 Hz.
    gyro_range    : Expected gyroscope range in deg/s. Used for gyro bias
                    correction. Set to 0 to disable.

    Returns corrected_accel (world-frame linear acceleration, gravity removed).
    """
    ahrs = imufusion.Ahrs()
    ahrs.settings = imufusion.Settings(
        imufusion.CONVENTION_NWU,   # North-West-Up world frame
        gain,
        gyro_range,
        10,                         # acceleration rejection threshold (deg)
        0,                          # magnetic rejection (unused, no magnetometer)
        int(5 * SAMPLE_RATE),       # recovery trigger period (5 seconds)
    )

    corrected = np.zeros_like(accel)
    for i in range(len(accel)):
        ahrs.update_no_magnetometer(gyro[i], accel[i], 1.0 / SAMPLE_RATE)
        corrected[i] = ahrs.earth_acceleration

    return corrected


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------

def residual_stats(data: np.ndarray) -> dict:
    """
    Compute mean, std, and mean vector magnitude for a corrected accel array.
    All in G. A perfect correction would give means of 0 and magnitude
    reflecting only true motion.
    """
    means = data.mean(axis=0)
    stds  = data.std(axis=0)
    mag   = np.sqrt((data ** 2).sum(axis=1)).mean()
    return {
        "x_mean": means[0], "y_mean": means[1], "z_mean": means[2],
        "x_std":  stds[0],  "y_std":  stds[1],  "z_std":  stds[2],
        "mag_mean": mag,
    }


def compare_methods(myogym: pd.DataFrame, gain: float, gyro_range: int,
                    gyro_thresh: float) -> pd.DataFrame:
    rows = []
    for pid in sorted(myogym["participant"].unique()):
        p     = myogym[myogym["participant"] == pid]
        accel = p[ACCEL_COLS].values.astype(float)
        gyro  = p[GYRO_COLS].values.astype(float)

        static_accel, grav_vec = remove_gravity_static(accel, gyro, gyro_thresh)
        quat_accel              = remove_gravity_quaternion(accel, gyro, gain, gyro_range)

        grav_mag = np.linalg.norm(grav_vec)

        for method, data in [("static", static_accel), ("quaternion", quat_accel)]:
            s = residual_stats(data)
            rows.append({
                "participant": pid,
                "method":      method,
                "grav_mag_G":  round(grav_mag, 4) if method == "static" else "—",
                **{k: round(v, 4) for k, v in s.items()},
            })

    return pd.DataFrame(rows)


def print_comparison(comp: pd.DataFrame) -> None:
    pivot_mag = comp.pivot(index="participant", columns="method", values="mag_mean")
    pivot_mag["improvement_G"] = (pivot_mag["static"] - pivot_mag["quaternion"]).round(4)
    pivot_mag = pivot_mag.round(4)

    static_rows = comp[comp["method"] == "static"].set_index("participant")

    print("=" * 70)
    print("GRAVITY VECTOR ESTIMATES (static method)")
    print("=" * 70)
    print(f"{'Participant':>12}  {'x':>8}  {'y':>8}  {'z':>8}  {'|mag|':>8}")
    for pid, row in static_rows.iterrows():
        print(f"{pid:>12}  {row['x_mean']+row['x_mean']*0:>8}  "  # placeholder — use grav_vec below
              f"  (see per-axis means below)")
    # Cleaner: just print the means table
    print()
    print(comp[comp["method"] == "static"][
        ["participant", "grav_mag_G", "x_mean", "y_mean", "z_mean"]
    ].to_string(index=False))

    print()
    print("=" * 70)
    print("RESIDUAL STATISTICS — mean should be ~0 for a good correction")
    print("=" * 70)
    for method in ["static", "quaternion"]:
        m = comp[comp["method"] == method]
        avg = m[["x_mean","y_mean","z_mean","x_std","y_std","z_std","mag_mean"]].mean()
        print(f"\n  {method.upper()}")
        print(f"    axis means (G): x={avg['x_mean']:+.4f}  "
              f"y={avg['y_mean']:+.4f}  z={avg['z_mean']:+.4f}")
        print(f"    axis stds  (G): x={avg['x_std']:.4f}  "
              f"y={avg['y_std']:.4f}  z={avg['z_std']:.4f}")
        print(f"    mean |mag| (G): {avg['mag_mean']:.4f}")

    print()
    print("=" * 70)
    print("PER-PARTICIPANT MEAN MAGNITUDE (lower = gravity better removed)")
    print("positive improvement = quaternion reduced residual magnitude")
    print("=" * 70)
    print(pivot_mag.to_string())
    print()
    winners = (pivot_mag["improvement_G"] > 0).sum()
    print(f"  Quaternion outperformed static on {winners}/{len(pivot_mag)} participants")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Remove gravity using Madgwick quaternion filter; compare with static subtraction."
    )
    parser.add_argument("--input",       required=True,  help="Path to input CSV")
    parser.add_argument("--output",      required=True,  help="Path to output CSV")
    parser.add_argument("--gain",        type=float, default=0.5,
                        help="Madgwick filter gain 0–1 (default: 0.5)")
    parser.add_argument("--gyro_range",  type=int,   default=200,
                        help="Expected gyroscope range in deg/s (default: 200)")
    parser.add_argument("--gyro_thresh", type=float, default=15.0,
                        help="Gyro magnitude threshold for static method (default: 15 deg/s)")
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"  {len(df)} rows, datasets: {df['dataset'].unique().tolist()}\n")

    myogym_mask = df["dataset"] == "myogym"
    myogym      = df[myogym_mask].copy()

    # --- Comparison ---
    print("Running comparison: static subtraction vs Madgwick quaternion filter ...\n")
    comp = compare_methods(myogym, args.gain, args.gyro_range, args.gyro_thresh)
    print_comparison(comp)

    # --- Apply quaternion method to produce output file ---
    df_out = df.copy()
    for pid in sorted(myogym["participant"].unique()):
        p_mask = myogym_mask & (df["participant"] == pid)
        p      = df[p_mask]
        accel  = p[ACCEL_COLS].values.astype(float)
        gyro   = p[GYRO_COLS].values.astype(float)

        corrected = remove_gravity_quaternion(accel, gyro, args.gain, args.gyro_range)
        df_out.loc[p_mask, ACCEL_COLS] = corrected

    df_out.to_csv(args.output, index=False)
    print(f"Saved quaternion-corrected data to {args.output}")
    print("(Other datasets are unchanged.)")


if __name__ == "__main__":
    main()
