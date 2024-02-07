import papermill as pm
import os

directory = "data/05022024_1a8_2b11_0.5nM/out/"

for file in os.listdir(directory):
    if file.endswith((".tif", ".tiff")):
        pm.execute_notebook(
            "01-fit-one-trajectory.ipynb",
            "temp_fit_notebook.ipynb",
            parameters=dict(in_path=directory, in_filename=file, out_path="out"),
        )
