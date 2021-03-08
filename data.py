# Author: Jeremie Gaffarel, lastly revised on the 17th of February 2021
# This module contains functions to get the concepts data (using open_file),
# and functions that configure all of the columns, as well as
# finally functions used to compute new columns
import open_file as OF


def comp_prop_mass(col_dry, col_wet, df):
	# Compute the propellant mass from the wet and dry mass
	return round(df[col_wet] - df[col_dry], 3)

# Declare each column configuration
def get_col_config():
	d_config = [
		OF.d_col("ID", str, weight=-1), # This column is required
		OF.d_col("Dry Mass [kg]", float),
		OF.d_col("Wet Mass [kg]", float, limit_value=15000, best=min, weight=5),
		OF.d_col("Thrust [kN]", float, is_list=True, best=max, weight=3),
		OF.d_col("Volume [m3]", float, is_list=True, limit_value=6, best=max, weight=5),
		OF.d_col("T [K]", str, range=[278, 303], weight=2),
		OF.d_col("P [bar]", float, range=[80, 150], weight=2),
		OF.d_col("TRL", int, best=max, weight=1),
		OF.d_col("Power [kW]", float, limit_value=20, best=min, weight=2),
		OF.d_col("EU", bool, best=True, weight=1)
		]
	# Define additional columns that will be computed later
	d_additional = [
		OF.d_col("Prop mass [kg]", float, best=max, operation=comp_prop_mass, op_param=("Dry Mass [kg]", "Wet Mass [kg]"), weight=3)
		]
	return d_config, d_additional

def get_data(fname, n_rows):
	"""
	Get the data from the file
	Inputs:
	 * fname (string): file name
	 * n_rows (int): number of rows to get from the file
	"""
	# Get the column config, and the column to be computed
	d_config, d_additional = get_col_config()
	return OF.import_data(fname, d_config, n_rows=n_rows, comp_col=d_additional)