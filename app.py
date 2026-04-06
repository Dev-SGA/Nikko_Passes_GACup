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
        ("PASS WON", 87.76, 22.38, 83.27, 41.83, "videos/P2.mp4"),
        ("PASS WON", 76.96, 24.54, 95.08, 22.71, None),
        ("PASS WON", 69.14, 18.06, 88.92, 13.40, "videos/P1.mp4"),
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
    end_opp_half = x_end >= 60
    start_opp_half = x_start >= 60

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

    progressive_total = int(df["progressive"].sum())
    progressive_successful = int(
        (df["progressive"] & df["type"].str.contains("WON", case=False)).sum()
    )
    progressive_accuracy = (
        progressive_successful / progressive_total * 100.0
        if progressive_total else 0.0
    )

    key_passes = int(df["video"].apply(has_video_value).sum())

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
        "progressive_successful_passes": progressive_successful,
        "progressive_accuracy_pct": round(progressive_accuracy, 2),
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
        is_progressive_success = bool(row["progressive"]) and not is_lost
        has_vid = has_video_value(row["video"])

        if is_lost:
            color = COLOR_FAIL
            alpha = 0.45
        elif is_progressive_success:
            color = COLOR_PROGRESSIVE
            alpha = 0.82
        else:
            color = COLOR_SUCCESS
            alpha = 0.75

        pitch.arrows(
            row["x_start"], row["y_start"],
            row["x_end"], row["y_end"],
            color=color,
            width=1.55,
            headwidth=2.25,
            headlength=2.25,
            ax=ax,
            zorder=3,
            alpha=alpha,
        )

        if has_vid:
            pitch.scatter(
                row["x_start"], row["y_start"],
                s=95,
                marker="o",
                facecolors="none",
                edgecolors="#FFD54F",
                linewidths=2.0,
                ax=ax,
                zorder=4,
            )

        pitch.scatter(
            row["x_start"], row["y_start"],
            s=START_DOT_SIZE,
            marker="o",
            color=color,
            edgecolors="white",
            linewidths=0.8,
            ax=ax,
            zorder=5,
            alpha=alpha,
        )

    ax.set_title(title, fontsize=12)

    legend_elements = [
        Line2D([0], [0], color=COLOR_SUCCESS, lw=2.5, label="Successful Pass"),
        Line2D([0], [0], color=COLOR_FAIL, lw=2.5, label="Unsuccessful Pass"),
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=2.5, label="Successful Progressive Pass (Opta)"),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="white",
            markersize=6,
            label="Start point (click)"
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="#FFD54F",
            markeredgewidth=2,
            markersize=7,
            label="Has video"
        ),
    ]

    legend = ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        shadow=False,
        fontsize="x-small",
        labelspacing=0.5,
        borderpad=0.5,
    )
    legend.get_frame().set_alpha(1.0)

    arrow = FancyArrowPatch(
        (0.45, 0.05),
        (0.55, 0.05),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=2,
        color="#333333",
    )
    fig.patches.append(arrow)

    fig.text(
        0.5,
        0.02,
        "Attack Direction",
        ha="center",
        va="center",
        fontsize=9,
        color="#333333",
    )

    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    img_obj = Image.open(buf)
    return img_obj, ax, fig

# ==========================
# Sidebar
# ==========================
st.sidebar.header("Match Selection")
selected_match = st.sidebar.radio("Choose the match", list(full_data.keys()), index=0)

st.sidebar.header("Pass Filter")
pass_filter = st.sidebar.radio(
    "Filter passes",
    ["All Passes", "Successful Only", "Unsuccessful Only", "Progressive Only"],
    index=0
)

df = full_data[selected_match].copy()

if pass_filter == "Successful Only":
    df = df[df["type"].str.contains("WON", case=False)].reset_index(drop=True)
elif pass_filter == "Unsuccessful Only":
    df = df[df["type"].str.contains("LOST", case=False)].reset_index(drop=True)
elif pass_filter == "Progressive Only":
    df = df[
        df["progressive"] & df["type"].str.contains("WON", case=False)
    ].reset_index(drop=True)

stats = compute_stats(df)

# ==========================
# Layout
# ==========================
col_stats, col_right = st.columns([1, 2], gap="large")

with col_stats:
    st.subheader("Statistics")

    # First shelf: totals only
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Passes", stats["total_passes"])
    c2.metric("Successful Passes", stats["successful_passes"])
    c3.metric("Accuracy", f'{stats["accuracy_pct"]:.1f}%')

    st.divider()

    # Second shelf: progressive passes
    st.subheader("Progressive Passes")
    p1, p2, p3 = st.columns(3)
    p1.metric("Total", stats["progressive_passes"])
    p2.metric("Successful", stats["progressive_successful_passes"])
    p3.metric("Accuracy", f'{stats["progressive_accuracy_pct"]:.1f}%')

    st.divider()

    st.subheader("Final Third")
    c7, c8, c9 = st.columns(3)
    c7.metric("Total", stats["final_third_total"])
    c8.metric("Successful", stats["final_third_success"])
    c9.metric("Unsuccessful", stats["final_third_unsuccess"])
    st.metric("Accuracy", f'{stats["final_third_accuracy_pct"]:.1f}%')

    st.divider()

    st.subheader("Passes Into the Box")
    d1, d2, d3 = st.columns(3)
    d1.metric("Total", stats["box_total"])
    d2.metric("Successful", stats["box_success"])
    d3.metric("Unsuccessful", stats["box_unsuccess"])
    st.metric("Accuracy", f'{stats["box_accuracy_pct"]:.1f}%')

with col_right:
    st.subheader("Pass Map (click the start dot)")

    img_obj, ax, fig = draw_pass_map(df, title=f"Pass Map - {selected_match}")
    click = streamlit_image_coordinates(img_obj, width=780)

    selected_pass = None

    if click is not None:
        real_w, real_h = img_obj.size
        disp_w, disp_h = click["width"], click["height"]

        pixel_x = click["x"] * (real_w / disp_w)
        pixel_y = click["y"] * (real_h / disp_h)

        mpl_pixel_y = real_h - pixel_y
        coords_clicked = ax.transData.inverted().transform((pixel_x, mpl_pixel_y))
        field_x, field_y = coords_clicked[0], coords_clicked[1]

        df_sel = df.copy()
        df_sel["dist"] = np.sqrt(
            (df_sel["x_start"] - field_x) ** 2 +
            (df_sel["y_start"] - field_y) ** 2
        )

        RADIUS = 7.0
        candidates = df_sel[df_sel["dist"] < RADIUS].copy()

        if not candidates.empty:
            candidates["has_video"] = candidates["video"].apply(has_video_value)
            candidates = candidates.sort_values(
                by=["has_video", "dist"],
                ascending=[False, True]
            )
            selected_pass = candidates.iloc[0]

    plt.close(fig)

    st.divider()
    st.subheader("Selected Event")

    if selected_pass is None:
        st.info("Click the start dot to inspect the pass details.")
    else:
        st.success(
            f"Selected pass: #{int(selected_pass['number'])} ({selected_pass['type']})"
        )
        st.write(
            f"Start: ({selected_pass['x_start']:.2f}, {selected_pass['y_start']:.2f})  \n"
            f"End: ({selected_pass['x_end']:.2f}, {selected_pass['y_end']:.2f})"
        )
        st.write(f"Progressive: {'Yes' if selected_pass['progressive'] else 'No'}")

        if has_video_value(selected_pass["video"]):
            try:
                st.video(selected_pass["video"])
            except Exception:
                st.error(f"Video file not found: {selected_pass['video']}")
        else:
            st.warning("No video is attached to this event.")
