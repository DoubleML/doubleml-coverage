# Includes Directory

This directory contains reusable Quarto components that can be included in the main documentation files.

## Files:

- `summary_table.qmd`: Generates the coverage results summary table
- `coverage_plot.qmd`: Creates the illustrative coverage plot

## Usage:

Include these files in your main documents using:
```
{{< include _includes/filename.qmd >}}
```