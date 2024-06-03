import os
import numpy as np
import platform
from pathlib import Path

def load_file(filename, subdir="iofiles", ndmin=1):
  """Loads data from a file in a subdirectory into a NumPy array of floats.

  Args:
      filename (str, optional): The name of the data file.
      subdir (str, optional): The name of the subdirectory containing the data. Defaults to "iofiles".

  Returns:
      A NumPy array containing the data from the file, or None if the file
      is not found or an error occurs during loading.
  """

  # Construct the full path to the file
  this_directory = os.path.dirname(__file__)
  full_subdir = os.path.join(this_directory, subdir)
  filepath = os.path.join(full_subdir, filename)
  
  try:
    data = np.loadtxt(filepath, dtype=float)
    return data
  except FileNotFoundError:
    print(f"Error: File '{filepath}' not found.")
    return None


def removeFE(datamat_j, obsmat_j):
	# Calculate the sum of observations for each individual
	obscounts = np.sum(obsmat_j, axis=1)

	# Calculate the sum of observed data for each individual
	obssums = np.sum(datamat_j * obsmat_j, axis=1)

	# Calculate fixed effects values
	FEvals = obssums / obscounts

	# Extend fixed effects values to the shape of the data matrix
	FEmtx = obsmat_j * FEvals[:, np.newaxis]

	# Remove fixed effects from the data matrix
	datamat_dm = datamat_j - FEmtx

	return datamat_dm, FEvals

if __name__ == "__main__":
    print(platform.system())
