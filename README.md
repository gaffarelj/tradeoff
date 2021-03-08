# Python Trade-off
This repository contains code to run trade-offs from and to Excel sheets, as well as a sensitivity analysis of the trade-off weights

## Trade-off setup
The Excel file name can be specified in main.py. It should have a similar format than the one used in the example xlsx file from this repository.

Mind that it is required that the Excel format is the same as the provided file (with a header, and only numerical values). The first column of the header has to be "ID" (capitalised). A range of value can be given in one or more cell by using a dash "val_low-val_high". The row will then be splitted in two: one with the lower values of all cells and one with the highest values, when splitting "-". The new rows IDs will be the old one, with "-L" or "-H" added.

The Excel file has to be closed for the code to run.

The column configuration is done in data.py. Please refer to open_file.d_col for all of the available configuration. If the weight is None, the column is not included in the tradeoff. A limitting value can be set for each column. For instance, if a parameter is required to be below 10 kg, the limit value is set to 10, and the "best" parameter set as min (as to give a higher score to low masses).

New columns can be set to be computed, this is done as an example in data.py, look for "d_additional". Mind that the column names are case sensitive.

## Data normalisation
The trade-off inputs can be a float, a boolean, or a range of values. They all all normalised, as to limit favorising very high or low values.

If the input is a float, this is done linearly between the best and worst of a column. If the worst is beyond the limit value, the latter is used as the worst instead, so that the bad concepts are not scored to extremely.

If the input is a boolean, 0 or 1 is returned.

If the input is a range, 1 is returned if any value within the range is within the specified limits. Otherwise, 0 is returned.

## Trade-off behaviour

If a value is unknown, it is replaced by the mean of the column. This can be changed in the "normalise" function in tradeoff.py

A sensitivity analysis can be run on the trade-off weights. No multiprocessing is available to make it faster, but the functions called during the runs have been optimised.

The tradeoff results can be saved to the same file in a new sheet, with the colors and formatting that can be expected from a trade-off table.

WARNING: if the results sheet called "Trade-off results" already exists, it is overwritten.

Columns with an std of 0 are automatically excluded from the tradeoff.

## Requirements

Required modules (tested on version):
pandas (1.2.2)
numpy (1.20.1)
openpyxl (3.0.6)
