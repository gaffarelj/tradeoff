# Author: Jeremie Gaffarel, lastly revised on the 4th of March 2021
# This file is the main code to get the concept data from a specified
# Excel file, run a tradeoff and its sensitivity analysis on the data,
# and save the tradeoff result in a new sheet of the same file

import data
import tradeoff as tc
import open_file as OF
import numpy as np

# Define the concept data file name
concepts_file = "example.xlsx"

# Import all concepts from the Excel sheet, computing new data
# Note: most of the configuration is hidden in the data.py module
concepts = data.get_data(concepts_file, 10)
# Get the column configurations
col_config, col_comp = data.get_col_config()
# Get the worst and best value of each column
worst_best = concepts.get_worst_best()
# Setup the tradeoff using the concepts data, the column configurations, and the best and worst values
tradeoff = tc.tradeoff(data = concepts.df, config=[*col_config, *col_comp], worst_best=worst_best)
# Run the sensitivity analysis for a std=1/3*mean, and for 10^4 samples
tradeoff.run_sensitivity(1e4, 1/3, box_plot=True)
# Save the tradeoff results in the same excel file, in a new sheet
tradeoff.get_output(format="excel", fname=concepts_file)