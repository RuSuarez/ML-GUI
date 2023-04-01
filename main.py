import tkinter as tk
from app import Application

# Create root window
root = tk.Tk()
root.title("Ruben - Machine Learning GUI")
app = Application(master=root)

# Start the application
app.mainloop()