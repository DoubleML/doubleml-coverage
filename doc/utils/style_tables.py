import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
from typing import Union, Optional, List, Any
from itables import show
from .styling import (
    TABLE_STYLING,
    COVERAGE_THRESHOLDS,
    get_coverage_tier_css_props,
)


# Define highlighting tiers using centralized color configuration
HIGHLIGHT_TIERS = [
    {"dist": COVERAGE_THRESHOLDS["poor"], "props": get_coverage_tier_css_props("poor")},
    {
        "dist": COVERAGE_THRESHOLDS["medium"],
        "props": get_coverage_tier_css_props("medium", "500"),
    },
    {"dist": COVERAGE_THRESHOLDS["good"], "props": get_coverage_tier_css_props("good")},
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
    # Use absolute difference to determine which tier applies
    abs_diff = np.abs(s_numeric - level)
    condition = abs_diff <= dist
    return np.where(condition, props, "")


def _determine_coverage_tier(value: float, level: float) -> str:
    """
    Determine which coverage tier a value belongs to based on distance from level.
    Returns the most specific (smallest distance) tier that applies.
    """
    if pd.isna(value):
        return ""

    abs_diff = abs(value - level)

    # Check tiers from most specific to least specific
    sorted_tiers = sorted(HIGHLIGHT_TIERS, key=lambda x: x["dist"])

    for tier in sorted_tiers:
        if abs_diff <= tier["dist"]:
            return tier["props"]

    return ""


def _apply_base_table_styling(styler: Styler) -> Styler:
    """
    Apply base styling to the table including headers, borders, and overall appearance.
    """
    # Define CSS styles for clean table appearance using centralized colors
    styles = [
        # Table-wide styling
        {
            "selector": "table",
            "props": [
                ("border-collapse", "separate"),
                ("border-spacing", "0"),
                ("width", "100%"),
                (
                    "font-family",
                    '"Segoe UI", -apple-system, BlinkMacSystemFont, "Roboto", sans-serif',
                ),
                ("font-size", "14px"),
                ("line-height", "1.5"),
                ("box-shadow", "0 2px 8px rgba(0,0,0,0.1)"),
                ("border-radius", "8px"),
                ("overflow", "hidden"),
            ],
        },
        # Header styling
        {
            "selector": "thead th",
            "props": [
                ("background-color", TABLE_STYLING["header_bg"]),
                ("color", TABLE_STYLING["header_text"]),
                ("font-weight", "600"),
                ("text-align", "center"),
                ("padding", "12px 16px"),
                ("border-bottom", f'2px solid {TABLE_STYLING["border"]}'),
                ("position", "sticky"),
                ("top", "0"),
                ("z-index", "10"),
            ],
        },
        # Cell styling
        {
            "selector": "tbody td",
            "props": [
                ("padding", "10px 16px"),
                ("text-align", "center"),
                ("border-bottom", f'1px solid {TABLE_STYLING["border"]}'),
                ("transition", "background-color 0.2s ease"),
            ],
        },
        # Row hover effect
        {
            "selector": "tbody tr:hover td",
            "props": [("background-color", TABLE_STYLING["hover_bg"])],
        },
        # Caption styling
        {
            "selector": "caption",
            "props": [
                ("color", TABLE_STYLING["caption_color"]),
                ("font-size", "16px"),
                ("font-weight", "600"),
                ("margin-bottom", "16px"),
                ("text-align", "left"),
                ("caption-side", "top"),
            ],
        },
    ]

    return styler.set_table_styles(styles)


def color_coverage_columns(
    styler: Styler, level: float, coverage_cols: list[str] = ["Coverage"]
) -> Styler:
    """
    Applies tiered highlighting to specified coverage columns of a Styler object.
    Uses non-overlapping logic to prevent CSS conflicts.
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

    # Apply base styling first
    current_styler = _apply_base_table_styling(styler)

    # Apply single tier styling to prevent conflicts
    def apply_coverage_tier_to_cell(s_col):
        """Apply only the most appropriate coverage tier for each cell."""
        return s_col.apply(lambda x: _determine_coverage_tier(x, level))

    current_styler = current_styler.apply(
        apply_coverage_tier_to_cell, subset=valid_coverage_cols
    )

    # Apply additional styling to coverage columns for emphasis
    current_styler = current_styler.set_properties(
        **{
            "text-align": "center",
            "font-family": "monospace",
            "font-size": "13px",
        },
        subset=valid_coverage_cols,
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
