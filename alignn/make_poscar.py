# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:46:42 2024

@author: haear
"""

"""Module to generate example dataset."""
from jarvis.core.atoms import Atoms
from tqdm import tqdm
import ast
import csv
from pathlib import Path

n_samples = 500

# Specify the path of the new folder
folder_path = Path("./poscar_data")

# Create the new folder
folder_path.mkdir(parents=True, exist_ok=True)

# Specify the path to your CSV file
csv_file_path = "alexandria_posc.csv"
with open(csv_file_path, mode='r') as file:
    # Create a CSV reader object
    csv_reader = csv.DictReader(file)
    
    # Iterate over each row in the CSV file
    for i, row in tqdm(enumerate(csv_reader)):
        
        if i>n_samples: break
        #print(row['mat_id'], row['lattice_mat'], row['composition'], row['coords'],'\n')
        
        atoms = Atoms(lattice_mat=ast.literal_eval(row['lattice_mat']),
                      coords=ast.literal_eval(row['coords']),
                      elements=ast.literal_eval(row['composition']),
                      cartesian=False)
        
        poscar_name = "./poscar_data/" + row['mat_id']
        atoms.write_poscar(poscar_name)
        #if i>3: break
            
file.close()