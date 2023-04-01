# Machine Learning GUI

This is a graphical user interface (GUI) for performing machine learning tasks on a CSV file. The user can upload a CSV file, select X and y columns, choose an algorithm, and perform regression. The user can also export predictions to a new CSV file.

## Dependencies

- Python 3.6+
- pandas
- tkinter
- matplotlib
- sklearn
- xgboost
- Pillow

## Usage

To run the application, simply run the following command:
python app.py

## GUI Description

- Upload CSV: Click this button to upload a CSV file.
- Clear Output: Click this button to clear the output text box.
- Select X columns: Use this list box to select the columns to use as X variables.
- Save: Click this button to save the selected X columns.
- Select y column: Use this list box to select the column to use as the y variable.
- Save: Click this button to save the selected y column.
- Select algorithm: Use this list of buttons to select the desired algorithm.
- Perform Regression: Click this button to perform the regression and display the results.
- Output text: This text box displays the output.
- Export Predictions: Click this button to export the predictions to a new CSV file.

## Methods

### load_csv()

Load a CSV file using the file dialog. Display the first 5 rows of the CSV file in the output text box. Populate the X and y list boxes with the column names.

### save_x_columns()

Save the selected X columns to the X columns attribute.

### save_y_column()

Save the selected y column to the y column attribute.

### set_logistic_regression()

Set the algorithm attribute to a LogisticRegression instance.

### set_decision_tree_classifier()

Set the algorithm attribute to a DecisionTreeClassifier instance.

### set_random_forest_classifier()

Set the algorithm attribute to a RandomForestClassifier instance.

### set_svm_classifier()

Set the algorithm attribute to an SVC instance.

### set_knn_classifier()

Set the algorithm attribute to a KNeighborsClassifier instance.

### set_xgboost_classifier()

Set the algorithm attribute to an XGBClassifier instance.

### clear_output()

Clear the output text box.

### perform_regression()

Perform the regression using the selected X and y columns and the selected algorithm. Display the precision, accuracy, recall, and F2 score in the output text box.

### export_predictions()

Perform the regression using the selected X and y columns and the selected algorithm. Add the predicted y and predicted y probability columns to the DataFrame. Save the DataFrame to a new CSV file using the file dialog. Display a message in the output text box indicating where the file was saved.
