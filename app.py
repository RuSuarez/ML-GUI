import tkinter as tk
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tkinter import filedialog
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from sklearn.metrics import classification_report


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
        self.algorithm = None
        self.df = None
        self.x_columns = []
        self.y_column = None
        self.master.geometry("1600x600")

    def create_widgets(self):
        # Create a frame for the output text
        self.output_frame = tk.Frame(self.master)
        self.output_frame.pack(side=tk.TOP, padx=10, pady=10)

        # Button to upload CSV file
        self.upload_button = tk.Button(self)
        self.upload_button["text"] = "Upload CSV"
        self.upload_button["command"] = self.load_csv
        self.upload_button.pack(side="top", anchor="w")

        # Clear Output button
        self.clear_output_button = tk.Button(self)
        self.clear_output_button["text"] = "Clear Output"
        self.clear_output_button["command"] = self.clear_output
        self.clear_output_button.pack(side="top", anchor="w")

        # Frame for X column selection
        self.x_frame = tk.Frame(self)
        self.x_frame.pack(side="left")

        # Label for X column selection
        self.x_label = tk.Label(self.x_frame)
        self.x_label["text"] = "Select X columns:"
        self.x_label.pack(side="top")

        # Frame for X column listbox and save button
        self.x_listbox_frame = tk.Frame(self.x_frame)
        self.x_listbox_frame.pack(side="top")

        # Listbox for X column selection
        self.x_listbox = tk.Listbox(self.x_listbox_frame, selectmode=tk.MULTIPLE)
        self.x_listbox.pack(side="left")

        # Button to save X columns
        self.save_x_button = tk.Button(self.x_listbox_frame)
        self.save_x_button["text"] = "Save"
        self.save_x_button["command"] = self.save_x_columns
        self.save_x_button.pack(side="left")

        # Select all X columns
        self.select_all_x_button = tk.Button(self.x_frame)
        self.select_all_x_button["text"] = "Select All"
        self.select_all_x_button["command"] = self.select_all_x_columns
        self.select_all_x_button.pack(side="top")

        # Frame for y column selection
        self.y_frame = tk.Frame(self)
        self.y_frame.pack(side="left")

        # Label for y column selection
        self.y_label = tk.Label(self.y_frame)
        self.y_label["text"] = "Select y column:"
        self.y_label.pack(side="top")

        # Frame for y column listbox and save button
        self.y_listbox_frame = tk.Frame(self.y_frame)
        self.y_listbox_frame.pack(side="top")

        # Listbox for y column selection
        self.y_listbox = tk.Listbox(self.y_listbox_frame)
        self.y_listbox.pack(side="left")

        # Button to save y column
        self.save_y_button = tk.Button(self.y_listbox_frame)
        self.save_y_button["text"] = "Save"
        self.save_y_button["command"] = self.save_y_column
        self.save_y_button.pack(side="left")

        # Frame for algorithm selection
        self.algorithm_frame = tk.Frame(self)
        self.algorithm_frame.pack(side="left")

        # Label for algorithm selection
        self.algorithm_label = tk.Label(self.algorithm_frame)
        self.algorithm_label["text"] = "Select algorithm:"
        self.algorithm_label.pack(side="top")

        # Button for Logistic Regression
        self.logistic_regression_button = tk.Button(self.algorithm_frame)
        self.logistic_regression_button["text"] = "Logistic Regression"
        self.logistic_regression_button["command"] = self.set_logistic_regression
        self.logistic_regression_button.pack(side="top")

        # Button for Decision Tree
        self.decision_tree_button = tk.Button(self.algorithm_frame)
        self.decision_tree_button["text"] = "Decision Tree"
        self.decision_tree_button["command"] = self.set_decision_tree_classifier
        self.decision_tree_button.pack(side="top")

        # Button for Random Forest
        self.random_forest_button = tk.Button(self.algorithm_frame)
        self.random_forest_button["text"] = "Random Forest"
        self.random_forest_button["command"] = self.set_random_forest_classifier
        self.random_forest_button.pack(side="top")

        # Button SVM
        self.svm_button = tk.Button(self.algorithm_frame)
        self.svm_button["text"] = "Support Vector Machine"
        self.svm_button["command"] = self.set_svm_classifier
        self.svm_button.pack(side="top")

        # Button for K-Nearest Neighbors
        self.knn_button = tk.Button(self.algorithm_frame)
        self.knn_button["text"] = "K-Nearest Neighbors"
        self.knn_button["command"] = self.set_knn_classifier
        self.knn_button.pack(side="top")

        # Perform XGBoost
        self.xgboost_button = tk.Button(self.algorithm_frame)
        self.xgboost_button["text"] = "XGBoost"
        self.xgboost_button["command"] = self.set_xgboost_classifier
        self.xgboost_button.pack(side="top")

        # Perform Regression button
        self.fit_model_button = tk.Button(self)
        self.fit_model_button["text"] = "Fit model"
        self.fit_model_button["command"] = self.fit_model
        self.fit_model_button.pack(side="top")

        # Output text
        self.output_text = tk.Text(self.output_frame, height=30, width=180)
        self.output_text.pack(side="top")

        # Export button
        self.export_button = tk.Button(self)
        self.export_button["text"] = "Export Predictions on Test data"
        self.export_button["command"] = self.export_predictions
        self.export_button.pack(side="top")

        # Apply model button
        self.apply_model_button = tk.Button(self)
        self.apply_model_button["text"] = "Apply Model"
        self.apply_model_button["command"] = self.apply_model
        self.apply_model_button.pack(side="top")

    def load_csv(self):
        # Load CSV file using file dialog
        file_path = filedialog.askopenfilename()
        self.df = pd.read_csv(file_path)

        # Clear the text box and add scrollbar
        self.output_text.delete("1.0", tk.END)
        self.output_text.configure(wrap="none")

        # Display the first 5 rows in the text box
        self.output_text.insert(tk.END, "Your csv has the following format:\n")
        self.output_text.insert(tk.END, self.df.head(5).to_string() + "\n")

        # Populate X and y listboxes with column names
        for column in self.df.columns:
            self.x_listbox.insert(tk.END, column)
            self.y_listbox.insert(tk.END, column)

        # Add a horizontal scrollbar to the output_text box
        self.output_text_xscrollbar = tk.Scrollbar(self.output_frame, orient=tk.HORIZONTAL)
        self.output_text_xscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.output_text_xscrollbar.config(command=self.output_text.xview)
        self.output_text.configure(xscrollcommand=self.output_text_xscrollbar.set)

        # Add the Select All button to the x_frame
        self.select_all_x_button = tk.Button(self.x_frame, text="Select All", command=self.select_all_x_columns)
        self.select_all_x_button.pack(side=tk.BOTTOM)


    def select_all_x_columns(self):
        self.x_listbox.selection_set(0, tk.END)

    def save_x_columns(self):
        # Save selected X columns
        self.x_columns = [self.x_listbox.get(idx) for idx in self.x_listbox.curselection()]
        self.output_text.insert(tk.END, f"Selected X columns: {', '.join(self.x_columns)}\n")

    def save_y_column(self):
        # Save selected y column
        self.y_column = self.y_listbox.get(self.y_listbox.curselection())
        self.output_text.insert(tk.END, f"Selected y column: {self.y_column}\n")

    def set_logistic_regression(self):
        self.algorithm = LogisticRegression(max_iter=1000)
        self.output_text.insert(tk.END, "Logistic Regression selected\n")

    def set_decision_tree_classifier(self):
        self.algorithm = DecisionTreeClassifier()
        self.output_text.insert(tk.END, "Decision Tree Classifier selected\n")

    def set_random_forest_classifier(self):
        self.algorithm = RandomForestClassifier()
        self.output_text.insert(tk.END, "Random Forest Classifier selected\n")

    def set_svm_classifier(self):
        self.algorithm = SVC(probability=True)
        self.output_text.insert(tk.END, "Support Vector Machine Classifier selected\n")

    def set_knn_classifier(self):
        self.algorithm = KNeighborsClassifier()
        self.output_text.insert(tk.END, "K-Nearest Neighbors Classifier selected\n")

    def set_xgboost_classifier(self):
        self.algorithm = XGBClassifier()
        self.output_text.insert(tk.END, "XGBoost Classifier selected\n")

    def clear_output(self):
        # Clear the output text
        self.output_text.delete("1.0", tk.END)
        # Clear the X and y listboxes
        self.x_listbox.delete(0, tk.END)
        self.y_listbox.delete(0, tk.END)
        # Reset the x_columns and y_column variables
        self.x_columns = []
        self.y_column = None

    def fit_model(self):
        if self.df is None:
            self.output_text.insert("end", "Please upload a CSV file\n")
            return

        if not self.x_columns:
            self.output_text.insert("end", "Please select at least one X column\n")
            return

        if self.y_column is None:
            self.output_text.insert("end", "Please select a y column\n")
            return

        if self.algorithm is None:
            self.output_text.insert("end", "Please select an algorithm\n")
            return

        # Split the data into X and y
        X = self.df[self.x_columns]
        y = self.df[self.y_column]

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Fit the algorithm to the training data
        self.algorithm.fit(self.X_train, self.y_train)

        # Make predictions on the testing data
        self.y_test_pred = self.algorithm.predict(self.X_test)
        self.y_test_pred_proba = self.algorithm.predict_proba(self.X_test)[:, 1]

        # Calculate metrics
        report = classification_report(self.y_test, self.y_test_pred, output_dict=True)
        precision = report['1']['precision']
        accuracy = report['accuracy']
        recall = report['1']['recall']
        f2_score = (5 * precision * recall) / (4 * precision + recall)

        # Print the metrics
        self.output_text.insert("end", f"Precision: {precision:.2f}\n")
        self.output_text.insert("end", f"Accuracy: {accuracy:.2f}\n")
        self.output_text.insert("end", f"Recall: {recall:.2f}\n")
        self.output_text.insert("end", f"F2 Score: {f2_score:.2f}\n")


    def export_predictions(self):
        if self.algorithm is None:
            self.output_text.insert("end", "Please select an algorithm\n")
            return

        # Make predictions on the test dataset
        y_pred_proba = self.algorithm.predict_proba(self.X_test)[:, 1]
        y_pred = self.algorithm.predict(self.X_test)

        # Create a new dataframe with the test data and predictions
        test_df = pd.concat([self.X_test, self.y_test], axis=1)
        test_df["predicted_y"] = y_pred
        test_df["predicted_y_proba"] = y_pred_proba

        # Open file dialog to save CSV file
        file_path = filedialog.asksaveasfilename(defaultextension=".csv")

        # Export the DataFrame to a CSV file
        test_df.to_csv(file_path, index=False)

        self.output_text.insert("end", f"Predictions exported to {file_path}\n")

    def apply_model(self):
        # Check if a model has been trained
        if self.algorithm is None:
            self.output_text.insert("end", "Please fit a model first\n")
            return
        
        # Load new CSV file
        file_path = filedialog.askopenfilename()
        new_df = pd.read_csv(file_path)
        
        # Get X columns from the new CSV
        new_X = new_df[self.x_columns]
        
        # Make predictions using the previously fit model
        new_y_pred_proba = self.algorithm.predict_proba(new_X)[:, 1]
        new_y_pred = self.algorithm.predict(new_X)
        
        # Add predictions to the new CSV
        new_df["predicted_y"] = new_y_pred
        new_df["predicted_y_proba"] = new_y_pred_proba
        
        # Open file dialog to save CSV file
        file_path = filedialog.asksaveasfilename(defaultextension=".csv")
        
        # Export the DataFrame to a CSV file
        new_df.to_csv(file_path, index=False)
        
        self.output_text.insert("end", f"Predictions exported to {file_path}\n")
