import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def find_csv_path(arg_path: str | None = None) -> Path:
	"""Resolve the path to housing.csv.

	Priority:
	1. If arg_path provided and exists, use it.
	2. Look next to this script (script directory).
	3. Look in the current working directory.
	Raises FileNotFoundError with a helpful message if not found.
	"""
	script_dir = Path(__file__).resolve().parent
	if arg_path:
		p = Path(arg_path)
		if not p.is_absolute():
			# allow relative to cwd
			p = (Path.cwd() / arg_path).resolve()
		if p.exists():
			return p
		# fall through to search locations below for nicer message

	candidates = [script_dir / 'housing.csv', Path.cwd() / 'housing.csv']
	for c in candidates:
		if c.exists():
			return c

	raise FileNotFoundError(
		"Could not find 'housing.csv'. Searched the following locations:\n"
		f" - next to the script: {candidates[0]}\n"
		f" - current working directory: {candidates[1]}\n"
		"Provide the full path as the first argument when running the script, e.g.:\n"
		"  python house_price.py C:\\path\\to\\housing.csv"
	)


def main():
	# Allow passing a path as the first CLI argument
	arg = sys.argv[1] if len(sys.argv) > 1 else None
	csv_path = find_csv_path(arg)

	# Load the data from CSV, specifying whitespace as the delimiter and no header
	data = pd.read_csv(csv_path, delim_whitespace=True, header=None)

	# Show the first few rows to understand it
	print(f"Loaded data from: {csv_path}\n")
	print(data.head())
	print(f"Data shape: {data.shape}")

	# Assume the last column is the target and the rest are features
	X = data.iloc[:, :-1]
	y = data.iloc[:, -1]
	print(f"X shape: {X.shape}")
	print(f"y shape: {y.shape}")

	# Split data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	print(f"X_train shape: {X_train.shape}")
	print(f"X_test shape: {X_test.shape}")

	# Create and train the linear regression model
	model = LinearRegression()
	model.fit(X_train, y_train)

	# Make predictions on the test set
	y_pred = model.predict(X_test)

	# Calculate and print the Mean Squared Error
	mse = mean_squared_error(y_test, y_pred)
	print(f"Mean Squared Error: {mse:.2f}")

	# Create a scatter plot to visualize predictions vs actual values
	plt.figure(figsize=(10, 6))
	plt.scatter(y_test, y_pred, alpha=0.7)
	plt.xlabel("Actual Prices")
	plt.ylabel("Predicted Prices")
	plt.title("Actual vs. Predicted Housing Prices")
	plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--') # Perfect prediction line
	plt.grid(True)
	plt.show()

if __name__ == '__main__':
	main()