import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================
# Page Configuration
# ==========================
st.set_page_config(layout="wide", page_title="Pass Map Dashboard (Interactive)")
st.title("Pass Map Dashboard")
st.caption("Click the start dot to select the pass event.")

# ==========================
# Configuration
# ==========================
FINAL_THIRD_LINE_X = 80

# Box dimensions for StatsBomb pitch
BOX_X_MIN = 102
BOX_Y_MIN = 18
BOX_Y_MAX = 62

# Goal center in StatsBomb coordinates
GOAL_X = 120
GOAL_Y = 40

# Progressive pass thresholds converted to StatsBomb pitch scale
# Approximation:
# 30m -> 24 units
# 15m -> 12 units
# 10m -> 8 units
PROG_OWN_HALF_THRESHOLD = 24
PROG_CROSS_HALF_THRESHOLD = 12
PROG_OPP_HALF_THRESHOLD = 8

# Colors
COLOR_SUCCESS = "#8E8E8E"       # gray
COLOR_FAIL = "#F2A3A3"          # very light red
COLOR_PROGRESSIVE = "#2F80ED"  # blue

# ==========================
# DATA
# ==========================
matches_data = {
    "Vs San Jose": [
        ("PASS WON", 85.27, 21.05, 93.25, 7.92, None),
        ("PASS WON", 103.55, 32.85, 108.04, 39.67, None),
        ("PASS WON", 86.26, 76.41, 63.99, 66.60, None),
    ],

    "Vs Copehagen": [
        ("PASS WON", 100.39, 1.10, 93.91, 5.76, None),
        ("PASS WON", 67.98, 61.28, 64.32, 51.80, None),
        ("PASS WON", 82.61, 54.79, 90.92, 69.26, None),
        ("PASS WON", 99.56, 65.77, 88.26, 57.95, None),
        ("PASS WON", 97.74, 69.09, 105.55, 72.58, None),

        ("PASS LOST", 45.70, 24.21, 57.34, 25.04, None),
        ("PASS LOST", 91.09, 29.03, 107.54, 25.70, None),
        ("PASS LOST", 86.60, 14.07, 104.88, 19.89, None),
        ("PASS LOST", 86.93, 51.64, 99.56, 55.13, None),
    ],

    "Vs Sporting": [
        ("PASS WON", 70.31, 35.01, 61.66, 46.98, None),
        ("PASS WON", 87.76, 22.38, 83.27, 41.83, None),
        ("PASS WON", 76.96, 24.54, 95.08, 22.71, None),
        ("PASS WON", 69.14, 18.06, 88.92, 13.40, None),
    ],
}

# ==========================
# Helpers
# ==========================
def has_video_value(v) -> bool:
    return pd.notna(v) and str(v).strip() != ""

def distance_to_goal(x, y):
    return np.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2)

def is_progressive_pass(x_start, y_start, x_end, y_end) -> bool:
    """
    Progressive pass using an Opta-style criterion based on distance reduction
    toward the opponent goal.

    Thresholds used:
    - own half to own half: at least 30m closer to goal
    - own half to opponent half: at least 15m closer to goal
    - opponent half to opponent half: at least 10m closer to goal

    Converted to StatsBomb scale:
    - 30m -> 24 units
    - 15m -> 12 units
    - 10m -> 8 units
    """
    start_dist = distance_to_goal(x_start, y_start)
    end_dist = distance_to_goal(x_end, y_end)
    gain = start_dist - end_dist

    start_own_half = x_start < 60
    end_own_half = x_end < 60
    start_opp_half = x_start >= 60
    end_opp_half = x_end >= 60

    if start_own_half and end_own_half:
        return gain >= PROG_OWN_HALF_THRESHOLD
    elif start_own_half and end_opp_half:
        return gain >= PROG_CROSS_HALF_THRESHOLD
    elif start_opp_half and end_opp_half:
        return gain >= PROG_OPP_HALF_THRESHOLD
    else:
        return False

# ==========================
# Build DataFrames
# ==========================
dfs_by_match = {}
for match_name, events in matches_data.items():
    dfm = pd.DataFrame(
        events,
        columns=["type", "x_start", "y_start", "x_end", "y_end", "video"]
    )
    dfm["number"] = np.arange(1, len(dfm) + 1)
    dfm["progressive"] = dfm.apply(
        lambda row: is_progressive_pass(
            row["x_start"], row["y_start"], row["x_end"], row["y_end"]
        ),
        axis=1
    )
    dfs_by_match[match_name] = dfm

df_all = pd.concat(dfs_by_match.values(), ignore_index=True)
full_data = {"All Matches": df_all}
full_data.update(dfs_by_match)

# ==========================
# Stats
# ==========================
def compute_stats(df: pd.DataFrame) -> dict:
    total_passes = len(df)
    successful = int(df["type"].str.contains("WON", case=False).sum())
    unsuccessful = int(df["type"].str.contains("LOST", case=False).sum())
    accuracy = (successful / total_passes * 100.0) if total_passes else 0.0

    key_passes = int(df["video"].apply(has_video_value).sum())
    progressive_total = int(df["progressive"].sum())

    in_final_third = df["x_end"] >= FINAL_THIRD_LINE_X
    final_third_total = int(in_final_third.sum())
    final_third_success = int((in_final_third & df["type"].str.contains("WON", case=False)).sum())
    final_third_unsuccess = int((in_final_third & df["type"].str.contains("LOST", case=False)).sum())
    final_third_accuracy = (final_third_success / final_third_total * 100.0) if final_third_total else 0.0

    to_box = (
        (df["x_end"] >= BOX_X_MIN) &
        (df["y_end"] >= BOX_Y_MIN) &
        (df["y_end"] <= BOX_Y_MAX)
    )
    box_total = int(to_box.sum())
    box_success = int((to_box & df["type"].str.contains("WON", case=False)).sum())
    box_unsuccess = int((to_box & df["type"].str.contains("LOST", case=False)).sum())
    box_accuracy = (box_success / box_total * 100.0) if box_total else 0.0

    return {
        "total_passes": total_passes,
        "successful_passes": successful,
        "unsuccessful_passes": unsuccessful,
        "accuracy_pct": round(accuracy, 2),
        "key_passes": key_passes,
        "progressive_passes": progressive_total,
        "final_third_total": final_third_total,
        "final_third_success": final_third_success,
        "final_third_unsuccess": final_third_unsuccess,
        "final_third_accuracy_pct": round(final_third_accuracy, 2),
        "box_total": box_total,
        "box_success": box_success,
        "box_unsuccess": box_unsuccess,
        "box_accuracy_pct": round(box_accuracy, 2),
    }

# ==========================
# Draw pass map
# ==========================
def draw_pass_map(df: pd.DataFrame, title: str):
    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#f5f5f5",
        line_color="#4a4a4a"
    )
    fig, ax = pitch.draw(figsize=(7.9, 5.3))
    fig.set_dpi(110)

    ax.axvline(x=FINAL_THIRD_LINE_X, color="#FFD54F", linewidth=1.2, alpha=0.25)

    START_DOT_SIZE = 45

    for _, row in df.iterrows():
        is_lost = "LOST" in row["type"].upper()
        is_progressive = bool(row["progressive"])
        has_vid = has_video_value(row["video"])

        if is_progressive:
            color = COLOR_PROGRESSIVE
            alpha = 0.82
        elif is_lost:
            color = COLOR_FAIL
            alpha = 0.45
        else:
            color = COLOR_SUCCESS
            alpha = 0.75

        pitch.arrows(
            row["x_start"], row["y_start"],
            row["x_end"], row["y_end"],
            color=color,
            width=1.***

