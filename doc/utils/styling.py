"""
Styling utilities for DoubleML Coverage tables and documentation.

This module provides helper functions for applying consistent styling
based on the centralized theme configuration.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import copy


def _load_theme_config() -> Dict[str, Any]:
    """Load theme configuration from YAML file."""
    config_path = Path(__file__).parent / "theme.yml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Load configuration once at module import
_THEME = _load_theme_config()

# Expose configuration for backward compatibility and direct access
COVERAGE_COLORS = _THEME["coverage_colors"]
TABLE_STYLING = _THEME["table_styling"]
COVERAGE_THRESHOLDS = _THEME["coverage_thresholds"]


def get_coverage_tier_css_props(tier: str, font_weight: str = "600") -> str:
    """
    Generate CSS properties string for a coverage performance tier.

    Args:
        tier: One of 'good', 'medium', 'poor'
        font_weight: CSS font-weight value

    Returns:
        CSS properties string for use with pandas Styler
    """
    if tier not in COVERAGE_COLORS:
        raise ValueError(
            f"Unknown tier '{tier}'. Must be one of: {list(COVERAGE_COLORS.keys())}"
        )

    colors = COVERAGE_COLORS[tier]
    return (
        f"color:{colors['text']};"
        f"background-color:{colors['background']};"
        f"border-left:4px solid {colors['border']};"
        f"font-weight:{font_weight};"
    )


def get_coverage_tier_html_span(tier: str, text: str = None) -> str:
    """
    Generate HTML span element with coverage tier styling for documentation.

    Args:
        tier: One of 'good', 'medium', 'poor'
        text: Text to display (defaults to tier description)

    Returns:
        HTML span element with inline styling
    """
    if tier not in COVERAGE_COLORS:
        raise ValueError(
            f"Unknown tier '{tier}'. Must be one of: {list(COVERAGE_COLORS.keys())}"
        )

    colors = COVERAGE_COLORS[tier]
    display_text = text or colors["description"]

    return (
        f'<span style="background-color: {colors["background"]}; '
        f'color: {colors["text"]}; '
        f"padding: 2px 5px; "
        f"border-radius: 3px; "
        f'border-left: 3px solid {colors["border"]};">'
        f"{display_text}</span>"
    )


def get_theme_config() -> Dict[str, Any]:
    """
    Get the complete theme configuration.

    Returns:
        Dictionary containing all theme settings
    """
    return copy.deepcopy(_THEME)
