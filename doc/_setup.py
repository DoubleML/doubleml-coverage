"""
Shared setup for all DoubleML Coverage simulation pages.
This file is executed once and makes imports/setup available to all pages.
"""

import numpy as np
import pandas as pd
from itables import init_notebook_mode
import os
import sys

# Add doc directory to path
doc_dir = os.path.abspath(os.getcwd())
if doc_dir not in sys.path:
    sys.path.append(doc_dir)

# Import styling utilities
from utils.style_tables import generate_and_show_styled_table

# Initialize itables for interactive tables
init_notebook_mode(all_interactive=True)
