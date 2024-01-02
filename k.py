import numpy as np

def print_validation_result(et, tc, wt, name_metrics=["HDis      ", "Sens      ", "Spec      ", "Dice"]):
  """
  Prints the Validation result with corresponding name metrics for three numpy arrays.

  Args:
    et: A numpy array containing the first row of data.
    tc: A numpy array containing the second row of data.
    wt: A numpy array containing the third row of data.
    name_metrics: A list of strings containing the corresponding name metrics for each column.

  Returns:
    None
  """
  # Check if the number of columns and name metrics match
  if len(et) != len(tc) != len(wt) != len(name_metrics):
    raise ValueError("The number of columns and name metrics must be equal.")

  # Print the Validation result with row names
  print("Validation result:")
  print("     ", *name_metrics)  # Print header with metrics names
  print("ET:  ", *et)  # Print ET row with values
  print("TC:  ", *tc)  # Print TC row with values
  print("WT:  ", *wt)  # Print WT row with values

# Example usage
et = np.array([9.19304596, 0.79736987, 0.99977756, 0.81850597])
tc = np.array([11.20203122, 0.86141576, 0.99978695, 0.8690264])
wt = np.array([29.92456626, 0.94453308, 0.99863205, 0.89638405])

print_validation_result(et, tc, wt)
