# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 04:59:21 2024

@author: haear
"""
from time import time
import json
import pandas as pd
import numpy as np
#from jarvis.core.atoms import Atoms

# Specify the path to your JSON file


# Specify the path to save the CSV file
csv_file_path = "alexandria_prop.csv"

pre_poscar_path = "alexandria_posc.csv"

        # Read the JSON file in chunks


# Iterate over the chunks and process each chunk
for i in range(1):
    num = "0" * (3-len(str(i))) + str(i)
    
    json_file_path = f'alexandria_{num}.json/alexandria_{num}.json'
    
    t = time()
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)
    print('JSON read!', time()-t)
    
    
    df =  pd.DataFrame(data['entries'])
    
    #extract infos for making poscar files
    t = time()
    energies = df['energy']
    print(time()-t)
    
    t = time()
    ids = df['data'].apply(lambda x: x['mat_id'])
    ids = [ "POSCAR-" + x + ".vasp" for x in ids]
    print(time()-t)

    
    lattice_mats = df['structure'].apply(lambda x: x['lattice']['matrix'])
    composition = df['structure'].apply(lambda x: [y['species'][0]['element'] for y in x['sites']])
    coords = df['structure'].apply(lambda x: [y['abc'] for y in x['sites']])
    print('Data stored!')

    
    # atoms = Atoms()
    
    
    # Create a DataFrame from the extracted values to export to csv
    extracted_data_df = pd.DataFrame({'mat_id': ids, 'energy': energies})
    poscar_data_df = pd.DataFrame({'mat_id': ids, 'lattice_mat': lattice_mats, 'composition': composition, 'coords': coords})
    print("Data has been written to", csv_file_path)
    
    # Append the DataFrame to the CSV file
    mode = "a" if i > 0 else "w"  # Use "w" mode for the first chunk, "a" mode for subsequent chunks
    header = False if i > 0 else True  # Write header only for the first chunk
    extracted_data_df.to_csv(csv_file_path, mode=mode, index=False, header=header)
    poscar_data_df.to_csv(pre_poscar_path, mode=mode, index=False, header=header)
    



