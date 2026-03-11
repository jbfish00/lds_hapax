# LDS Hapax Legomena Analysis

An NLP project analyzing **hapax legomena** — words that appear exactly once in a text — within the **Book of Mormon**. The project investigates whether the distribution of hapax legomena across different narrative events can provide statistical evidence about authorship patterns and textual uniqueness.

## Background

Hapax legomena are commonly used in computational linguistics and biblical scholarship as a stylometric feature. This project applies that lens to the Book of Mormon, examining whether different narrative sections (events) differ significantly in their density of unique vocabulary, and whether those sections are *written* or *spoken* in origin.

## Analysis

- Identifies all hapax legomena across the full Book of Mormon text
- Locates each hapax in its specific verse, event, and authorship context
- Compares hapax density between *written* vs. *non-written* narrative events
- Uses Mann-Whitney U test, chi-squared test, and t-test for statistical comparison
- Visualizes distributions with boxplots and bar charts

## Files

| File | Description |
|------|-------------|
| `hapax_proj.py` | Main Python analysis script |
| `hapax_find_by_verse.R` | R script for verse-level hapax lookup |
| `BOM.csv` / `new_BOM.csv` | Book of Mormon text with verse metadata |
| `hapax_legomena_details.csv` | All identified hapax legomena with verse/event context |

## Technologies

- Python, NLTK, Pandas, Matplotlib, Seaborn, SciPy
- R (supplementary verse-level analysis)
