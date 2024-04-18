# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 07:08:40 2024

@author: haear
"""

from time import time
import json
import pandas as pd
import numpy as np
#from jarvis.core.atoms import Atoms

# Specify the path to your JSON file


# Specify the path to save the CSV file
csv_file_path = "id_prop.csv"

pre_poscar_path = "alexandria_posc.csv"

        # Read the JSON file in chunks

total_samples = 100000
total_json_files = 45
grab_per_file = total_samples//total_json_files

grab_per_file = 10

final_str = ""

# Iterate over the chunks and process each chunk
for i in range(2):#total_json_files):
    num = "0" * (3-len(str(i))) + str(i)
    
    json_file_path = f'alexandria_{num}.json'
    
    t = time()
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)
    print('JSON read!', time()-t)
    
    t = time()
    df =  pd.DataFrame(data['entries'])
    
    #grab random samples
    rand_vec = np.random.choice(len(df), grab_per_file, replace=False)
    #print(rand_vec)
    df = df.iloc[rand_vec]
    #print(len(df))
    
    #extract infos for making poscar files

    

    def make_atoms(x):
        return {
            "lattice_mat": x['lattice']['matrix'],
            "coords": [y['abc'] for y in x['sites']],
            "elements": [y['species'][0]['element'] for y in x['sites']],
            "abc": [x['lattice'][i] for i in ['a', 'b', 'c']],
            "angles": [x['lattice'][i] for i in ['alpha', 'beta', 'gamma']],
            "cartesian": False,
            "props": ["", "", "", "", "", "", "", "", "", ""]
            }
    
    jid = df['data'].apply(lambda x: x['mat_id'])
    atoms = df['structure'].apply(lambda x: make_atoms(x))
    total_energy = df['energy']/df['data'].apply(lambda x: x['nsites']) #name is confusing but this is energy/atom
    forces = df['structure'].apply(lambda x: [y['properties']['forces'] for y in x['sites']])
    stresses = df['data'].apply(lambda x: x['stress'])
    
    print('Data stored!', time()-t)

    
    # atoms = Atoms()
    
    
    # Create a DataFrame from the extracted values to export to csv
    extracted_data_df = pd.DataFrame({'jid': jid,
                                      'atoms': atoms,
                                      "total_energy": total_energy,
                                      'forces': forces,
                                      'stresses': stresses})
    
    extracted_json = extracted_data_df.to_json(orient='records')
    
    if len(final_str)==0: 
        final_str += extracted_json
        
    else:
        final_str = final_str[:-1] + "," + extracted_json[1:]
    
with open("data_test_ff.json", "w") as f:
    f.write(final_str)



