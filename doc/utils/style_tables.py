import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
from typing import Union, Optional, List, Any
from itables import show


# Define highlighting tiers as a list of dictionaries or tuples
# Each element defines: dist, props. Applied in order (later rules can override).
# Order: from least specific (largest dist) to most specific (smallest dist)
# or ensure the _apply_highlight_range logic correctly handles overlaps if props are different.
# Current logic: more specific (smaller dist) rules are applied last and override.
HIGHLIGHT_TIERS = [
    {"dist": 1.0, "props": "color:black;background-color:red;"},
    {"dist": 0.1, "props": "color:black;background-color:yellow;"},
    {"dist": 0.05, "props": "color:white;background-color:darkgreen;"},
]


def _apply_highlight_range(
    s_col: pd.Series, level: float, dist: float, props: str
) -> np.ndarray:
    """
    Helper function for Styler.apply. Applies CSS properties based on a numeric range.
    Returns an array of CSS strings.
    """
    s_numeric = pd.to_numeric(
        s_col, errors="coerce"
    )  # Convert to numeric, non-convertibles become NaN
    # Apply style ONLY if value is WITHIN the current dist from level
    # This means for tiered styling, the order of applying styles in the calling function matters.
    # If a value falls into multiple dist categories, the LAST applied style for that dist will win.
    condition = (s_numeric >= level - dist) & (s_numeric <= level + dist)
    return np.where(condition, props, "")


def color_coverage_columns(
    styler: Styler, level: float, coverage_cols: list[str] = ["Coverage"]
) -> Styler:
    """
    Applies tiered highlighting to specified coverage columns of a Styler object.
    The order of application matters: more specific (narrower dist) rules are applied last to override.
    """
    if not isinstance(styler, Styler):
        raise TypeError("Expected a pandas Styler object.")

    # Ensure coverage_cols is a list
    if isinstance(coverage_cols, str):
        coverage_cols = [coverage_cols]

    # Filter for columns that actually exist in the DataFrame being styled
    valid_coverage_cols = [col for col in coverage_cols if col in styler.data.columns]

    if not valid_coverage_cols:
        return styler  # No valid columns to style

    # Apply highlighting rules from the defined tiers
    # The order in HIGHLIGHT_TIERS is important if props are meant to override.
    # Pandas Styler.apply applies styles sequentially. If a cell matches multiple
    # conditions from different .apply calls, the styles from later calls typically override
    # or merge with earlier ones, depending on the CSS properties.
    # For background-color, later calls will override.
    current_styler = styler
    for tier in HIGHLIGHT_TIERS:
        current_styler = current_styler.apply(
            _apply_highlight_range,
            level=level,
            dist=tier["dist"],
            props=tier["props"],
            subset=valid_coverage_cols,
        )

    # Set font to bold for the coverage columns
    current_styler = current_styler.set_properties(
        **{"font-weight": "bold"}, subset=valid_coverage_cols
    )
    return current_styler


def create_styled_table(
    df: pd.DataFrame,
    level: float,
    n_rep: Union[int, str],
    caption_prefix: str = "Coverage",
    coverage_cols: List[str] = ["Coverage"],
    float_precision: str = "{:.3f}",
) -> Styler:
    """
    Creates a styled pandas DataFrame (Styler object) for display.
    - Hides the DataFrame index.
    - Formats float columns to a specified precision.
    - Applies conditional highlighting to coverage columns.
    - Sets a descriptive caption.
    """
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame({"Error": ["Input is not a DataFrame."]}).style.hide(
            axis="index"
        )

    if df.empty:
        empty_df_cols = df.columns if df.columns.tolist() else ["Info"]
        message_val = (
            ["No data to display."]
            if not df.columns.tolist()
            else [None] * len(empty_df_cols)
        )
        df_to_style = pd.DataFrame(
            (
                dict(zip(empty_df_cols, [[v] for v in message_val]))
                if not df.columns.tolist()
                else {}  # Pass empty dict for empty DataFrame with columns
            ),
            columns=empty_df_cols,
        )
        return df_to_style.style.hide(axis="index").set_caption("No data to display.")

    # Prepare float formatting dictionary
    float_cols = df.select_dtypes(include=["float", "float64", "float32"]).columns
    format_dict = {col: float_precision for col in float_cols if col in df.columns}

    # Create and set the caption text
    level_percent = level * 100
    if abs(level_percent - round(level_percent)) < 1e-9:
        level_display = f"{int(round(level_percent))}"
    else:
        level_display = f"{level_percent:.1f}"

    n_rep_display = str(n_rep)  # Ensure n_rep is a string for the caption

    caption_text = f"{caption_prefix} for {level_display}%-Confidence Interval over {n_rep_display} Repetitions"

    # Chain Styler methods
    styled_df = (
        df.style.hide(axis="index")
        .format(
            format_dict if format_dict else None
        )  # Pass None if no float cols to format
        .pipe(color_coverage_columns, level=level, coverage_cols=coverage_cols)
        .set_caption(caption_text)
    )

    return styled_df


def generate_and_show_styled_table(
    main_df: pd.DataFrame,
    filters: dict[str, Any],
    display_cols: List[str],
    n_rep: Union[int, str],
    level_col: str = "level",
    rename_map: Optional[dict[str, str]] = None,
    caption_prefix: str = "Coverage",
    coverage_highlight_cols: List[str] = ["Coverage"],
    float_precision: str = "{:.3f}",
):
    """
    Filters a DataFrame based on a dictionary of conditions,
    creates a styled table, and displays it.
    """
    if main_df.empty:
        print("Warning: Input DataFrame is empty.")
        # Optionally, show an empty table or a message
        empty_styled_df = (
            pd.DataFrame(columns=display_cols)
            .style.hide(axis="index")
            .set_caption("No data available (input empty).")
        )
        show(empty_styled_df, allow_html=True)
        return

    # Build filter condition
    current_df = main_df
    filter_conditions = []
    filter_description_parts = []

    for col, value in filters.items():
        if col not in current_df.columns:
            print(
                f"Warning: Filter column '{col}' not found in DataFrame. Skipping this filter."
            )
            continue
        current_df = current_df[current_df[col] == value]
        filter_conditions.append(f"{col} == {value}")
        filter_description_parts.append(f"{col}='{value}'")

    filter_description = " & ".join(filter_description_parts)

    if current_df.empty:
        level_val = filters.get(level_col, "N/A")
        level_percent_display = (
            f"{level_val*100}%" if isinstance(level_val, (int, float)) else level_val
        )
        caption_msg = f"No data after filtering for {filter_description} at {level_percent_display} level."
        print(f"Warning: {caption_msg}")
        empty_styled_df = (
            pd.DataFrame(columns=display_cols)
            .style.hide(axis="index")
            .set_caption(caption_msg)
        )
        show(empty_styled_df, allow_html=True)
        return

    df_filtered = current_df[
        display_cols
    ].copy()  # Select display columns after filtering

    if rename_map:
        df_filtered.rename(columns=rename_map, inplace=True)

    # Determine the level for styling from the filters, if present
    styling_level = filters.get(level_col)
    if styling_level is None or not isinstance(styling_level, (float, int)):
        print(
            f"Warning: '{level_col}' not found in filters or is not numeric. Cannot determine styling level for highlighting."
        )
        # Fallback or raise error, for now, we'll proceed without level-specific caption part if it's missing
        # Or you could try to infer it if there's only one unique level in the filtered data
        if level_col in df_filtered.columns and df_filtered[level_col].nunique() == 1:
            styling_level = df_filtered[level_col].iloc[0]
        else:  # Default to a common value or skip styling that depends on 'level'
            styling_level = 0.95  # Default, or handle error

    styled_table = create_styled_table(
        df_filtered,
        styling_level,  # Use the level from filters for styling
        n_rep,
        caption_prefix=caption_prefix,
        coverage_cols=coverage_highlight_cols,
        float_precision=float_precision,
    )
    show(styled_table, allow_html=True)
