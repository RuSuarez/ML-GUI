![Screenshot](https://user-images.githubusercontent.com/121521393/229355464-5dd6c250-08a2-4174-87ee-a51c1f076669.png)

# Machine Learning GUI

This is a Graphical User Interface (GUI) for fitting and applying classification models on CSV files. The GUI allows the user to load a CSV file, select X and y columns, choose a classification algorithm, fit the model, evaluate the performance metrics, export the predictions, and apply the model to a new CSV file.

Example data included in repo.
- txs_train_test.csv for train and test model
- new_txs.csv to apply model to "new" transations 

Data source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Dependencies

- Python 3.6+
- pandas
- tkinter
- matplotlib
- sklearn
- xgboost

## Usage

To run the application, simply run the following command:
python app.py

## GUI Description

- Upload a CSV file by clicking the "Upload CSV" button.
- Select features by clicking on the column names in the "Select X columns" listbox and clicking the "Save" button.
- If one or more feature columns contains null values, I pop up window will show offering some options to handle said missing values.
- Select the target column by clicking on the column name in the "Select y column" listbox and clicking the "Save" button.
- Select an algorithm by clicking on the corresponding button in the "Select algorithm" section. The available algorithms are:
    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier
    - Support Vector Machine Classifier
    - K-Nearest Neighbors Classifier
    - XGBoost Classifier
- Click the "Fit model" button to fit the algorithm to the data and calculate metrics.
- Click the "Export Predictions on Test data" button to export a new CSV file with the predictions appended to the test data.
- Click the "Apply Model" button to read a new CSV file and automatically fit the model and export a new CSV with the predictions appended to it, based on the model previously fit. In case no model has been fit yet the GUI will print a message that a model needs to be trained first.
- Use the "Clear Output" button to clear the output text box.
