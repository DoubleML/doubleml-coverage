import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler


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
    n_rep: int,  # Or Union[int, str] if "N/A" is possible
    caption_prefix: str = "Coverage",
    coverage_cols: list[str] = ["Coverage"],
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
        return (
            pd.DataFrame(
                (
                    dict(zip(empty_df_cols, [[v] for v in message_val]))
                    if not df.columns.tolist()
                    else []
                ),
                columns=empty_df_cols,
            )
            .style.hide(axis="index")
            .set_caption("No data to display.")
        )

    # Prepare float formatting dictionary
    float_cols = df.select_dtypes(include=["float"]).columns
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
