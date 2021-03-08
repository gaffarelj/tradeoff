# Author: Jeremie Gaffarel, lastly revised on the 17th of February 2021
# This module contains a function to get data from an Excel file, 
# pre-processed for a tradeoff.
import pandas as pd
import numpy as np
import sys


class import_data:
	"""
	Class used to read and pre-process the Excel sheet and its data
	that will later be used for the tradeoff
	Inputs:
	 * fname (string): Excel filename
	 * col_config (list[open_file.d_col]): configuration of each of the column
	 * n_rows (int): number of rows from which data will be gathered from the sheet
	 * comp_col (list[open_file.d_col]): 
	"""
	def __init__(self, fname, col_config, n_rows=None, comp_col=None):
		# Read the Excel file containing the data
		try:
			self.df = pd.read_excel(fname, nrows=n_rows)
		except PermissionError:
			print("Error: please close the Excel file before running this code.")
			sys.exit()
		# Save the column configuration
		self.col_config = col_config
		# Save the new column to compute
		self.comp_col = comp_col
		# Split the rows that contains multiple values,
		# and apply the correct type to every value
		self.split_and_type()
		# If columns to be computed are specified, do so
		if comp_col is not None:
			self.comp_new()

	def split_and_type(self):
		"""
		Split the low and high options when they are mixed in a single row.
		At the same type, apply the correct type to every value in the DataFrame.
		"""
		# Create a temporary list of rows
		temp_df = []
		# Go trough every row of the DataFrame
		for _i, row in self.df.iterrows():
			# Define a boolean to save whether the row has been split or not
			split_row = False
			# Define the low and high part of the split row by copying the current one, even if it is not split
			row_low, row_high = row.copy(), row.copy()
			# Change the ID of the low and high part of the potentially split row
			row_low["ID"], row_high["ID"] = str(row_low["ID"]) + "-L", str(row_high["ID"]) + "-H"
			# Go trough each column of the individual row
			for col in self.df:
				# Go trough each column specified configuration
				for config in self.col_config:
					# If the config column match, and this column can be split...
					if config.name == col and config.is_list:
						# Split it at the dash, making sure every list element is the correct type
						row_split = [config.dtype(val) for val in str(row[col]).split("-")]
						# If the split results in more than a single element...
						if len(row_split) > 1:
							# Remember than the split was useful
							split_row = True
							# Change the low and high value in the two split rows
							# using the min and max from the split list
							row_low[col], row_high[col] = min(row_split), max(row_split)
						# If there is only a single element, replace it as the row (to save the applied dtype)
						else:
							row[col] = row_split[0]
					# If the config column match and the value is not a NaN...
					elif config.name == col and not pd.isna(row[col]) and col != "ID":
						# Convert the value to the correct type
						val = row[col]
						row[col], row_low[col], row_high[col] = val, val, val
			# If the split happened, add the low and high split rows to the temporary list of rows
			if split_row:
				temp_df.append(row_low), temp_df.append(row_high)
			# If there was no split, simply save the row with elements of the correct type
			else:
				temp_df.append(row)
		# Convert the temporary list as the DataFrame
		self.df = pd.DataFrame(temp_df).reset_index(drop=True)

	def get_worst_best(self):
		"""
		Return the best and worst value of each column
		"""
		# Declare the empty dictionary that will contain the worst/best values,
		# and the keys as the column names
		worst_best = dict()
		# Go trough each column in the column configurations
		for col_config in [*self.col_config, *self.comp_col]:
			# Extract the column name
			col_name = col_config.name
			# If the best value of the column is either minimum or maximum...
			if col_config.best in [min, max]:
				# If the best is the maximum
				if col_config.best == max:
					# Set the worst value as the min, and the best as the max, filter out the nan
					worst, best = np.nanmin(self.df[col_name]), np.nanmax(self.df[col_name])
				else:
					# Set the worst value as the max, and the best as the min, filter out the nan
					worst, best = np.nanmax(self.df[col_name]), np.nanmin(self.df[col_name])
				# If the column contains a limitting value, set it as the worst one
				if col_config.limit_value is not None:
					worst = col_config.limit_value
				# Save the worst and best value in the dict, with the key as the column name
				worst_best[col_name] = (worst, best)
			else:
				# If the best is a single value, set it as the "worst best"
				worst_best[col_name] = col_config.best
		return worst_best

	def comp_new(self):
		"""
		Compute new columns from the existing ones
		"""
		# Loop trough the column that shall be added
		for add_col in self.comp_col:
			# Add the column to the DataFrame, computing it using the defined function
			self.df[add_col.name] = add_col.operation(*add_col.op_param, self.df)
		

class d_col:
	"""
	Class used to specify how each column in the data frame is configured
	Inputs:
	 * name (str): name of the column in the Excel sheet
	 * dtype (type): data type used in the column
	 * is_list (bool): whether the data in the specific column can be a list or not (as "val_low-val_high")
	 * limit_value (float): the limiting value for the comlumn. For instance: if the mass shall be < 100 g, or the Impulse above > 300 Ns
	 * best (bool or min/max): specify if its best for the value to be maximised, minimised or a specified boolean
	 * operation (function): for additional columns that are not present in the Excel sheet, but will be computed based on it
	 * op_param (list): if operation is defined, list of parameters to used for it
	 * range (list): specify the range of value allowed for the column
	 * weight (float): weight of the column in the tradeoff (use -1 for the product name)
	"""
	def __init__(self, name, dtype, is_list=False, limit_value=None, best=None, \
		operation=None, op_param=None, range=None, weight=None):
		# Save all of the class inputs
		self.name = name
		self.dtype = dtype
		self.is_list = is_list
		self.best = best
		self.limit_value = limit_value
		self.operation = operation
		self.op_param = op_param
		self.range = range
		self.weight = weight
		self.color = []

	def set_colors(self, colors):
		"""
		Set the color of the column cells based on their normalised values
		Input: colors (list[tradeoff.color]): ordered list of colors to use for the tradeoff result table
		"""
		# Asses if the current column is included in the tradeoff
		if self.in_to():
			# Go trough each value of the column
			for val in self.val_norm:
				# If the value is one (corner case), set the color as the highest one in the list
				if val == 1:
					self.color.append(colors[-1])
				else:
					# Set the color as the corresponding one in the list of colors
					self.color.append(colors[int(val * len(colors))])

	def in_to(self):
		"""
		Return True if the column is to be included in the tradeoff,
		which is when the weight is not None and above 0
		"""
		return self.weight is not None and self.weight >= 0