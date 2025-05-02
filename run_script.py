#/home/k64728/Thesis_Repo/run_script.py
import papermill as pm


notebooks = ["mainImages.ipynb"]

for nb in notebooks:
    pm.execute_notebook(nb, f"output_{nb}", log_output=True)