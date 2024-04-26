import subprocess
import re
import os
import numpy as np
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
import torch
import shutil
from tqdm import tqdm
import ast
import glob

folder_path = r"C:\Users\haear\Documents\polytechnique\stage3a\scienti\eval_energy\DFT\TaVSb"

print("Reading SER data..")
#ser_file = r"C:\Users\haear\Documents\polytechnique\stage3a\scienti\eval_energy\DFT\TaFeSn\TaFeSn_corrected_uniques\DONE\SER-400.out"
ser_file = os.path.join(folder_path, "SER-400.out")
ser_data = {}

#Make a dict with SER data (reference energies)
with open(ser_file) as f:
  lines = f.read().splitlines()
  for line in lines[1:]:
    line_data = line.split()
    ser_data[line_data[0]] = line_data[3]

#compute the SER energy for a given compound
def compute_ser(content):
  num_list = []
  atom_list = []
  start = False
  for x in content:
      if start:
          if x == "|": break
          if x.isdigit(): num_list.append(float(x))
          else: atom_list.append(x)

      if x == "|": start = True

  tot = 0
  for i in range(len(num_list)):
    tot += num_list[i] * float(ser_data[atom_list[i]])
  return tot, sum(num_list)

"""
print("Cu:", ser_data["Cu"], "Si:", ser_data["Si"], "K_pv:",ser_data["K_pv"])
print("Ta:", ser_data["Ta"], "Fe:", ser_data["Fe"], "Sn:",ser_data["Sn"])
print("Ta:", ser_data["Ta"], "V:", ser_data["V"], "Sb:",ser_data["Sb"])
"""
print("Reading DFT data..")
#summary_file = r"C:\Users\haear\Documents\polytechnique\stage3a\scienti\eval_energy\DFT\TaFeSn\TaFeSn_corrected_uniques\DONE\SUMMARY"
summary_file = os.path.join(folder_path, "SUMMARY")

dft_data = {}

mode = 2 #change for different data formats

with open(summary_file) as f:
  lines = f.read().splitlines()

  for l in lines:
    line_data = l.split()

    if mode == 1: #JC data 1
        if len(line_data) == 19: #filter bugged calcs
            ser, n_at = compute_ser(line_data)
            syst, name, ref_energy = line_data[0], line_data[1], (line_data[-4], ser, n_at)
            #n_atoms = num_atoms(line_data)

    elif mode == 2: #Arsen data 1
        syst = "xxx"
        name, ref_energy = line_data[0], (line_data[1], line_data[1] - line_data[2], 1) #all energies are mean

    if syst not in dft_data.keys():
      dft_data[syst] = {name: ref_energy}
    else: dft_data[syst][name] = ref_energy
    
    

#pattern = r'\[(.*?)\]'
pattern = r'\[([-?\de\.,\s]+)\]'
#pattern = r"Energy\(eV\) \((-?\d+\.\d+),"

def extract_value(stdout):
  #return re.findall(pattern, stdout.strip().splitlines()[-1])[0]
  lines = stdout.splitlines()
  for line in lines:
    match = re.search(pattern, line)
    if match:
        value = match.group(1)
        return value
  return "NA"

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

txt_file_path =  r"C:\Users\haear\Documents\polytechnique\stage3a\scienti\eval_energy\DFT\TaVSb\data_alignn_etot.txt"
def save_to_drive(data):

  # Writing to txt file
  np.savetxt(txt_file_path, data, delimiter=',', fmt="%s")

  print("Array saved")

print("Reading inference..")

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

model_path = r"C:\Users\haear\Documents\polytechnique\stage3a\scienti\eval_energy\models\alex_random_1"
multi = False
force = False
mean_energy_calc = True
task="unrelaxed_energy"
relax_file = "CONTCAR"
base_file = "POSCAR"
data_energ = [("System", "Configuration", "Energy relaxed (dft)", "Energy non relax (ML)",  "Energy relax (ML)")]
count = 0

temp_path = r"C:\Users\haear\Documents\polytechnique\stage3a\scienti\eval_energy\DFT"
temp_base_path = os.path.join(temp_path, "temp_eval_energ_base")
temp_relax_path = os.path.join(temp_path, "temp_eval_energ_relax")
os.makedirs(temp_base_path)
os.makedirs(temp_relax_path)
name_storage = []

#Loop over all systems with energy calculated by DFT (see cell above)
# Loop over the 1st lvl subfolders in the folder
for foldname in tqdm(os.listdir(folder_path)): #foldname = 'Ta2V1Sb1_230' typically

    subfold_path = os.path.join(folder_path, foldname)
    if os.path.isdir(subfold_path) and foldname in dft_data.keys():

      # Loop over 2nd lvl subfolders in the 1st subfolder
      for system_fold in os.listdir(subfold_path):
        sysfold_path = os.path.join(subfold_path, system_fold)
        if os.path.isdir(sysfold_path) and system_fold in dft_data[foldname].keys(): #system_fold = 'Sb-Ta-V-7.vasp' typically
            
                shutil.copy(os.path.join(sysfold_path,base_file), os.path.join(temp_base_path, f"{count}"))
                shutil.copy(os.path.join(sysfold_path,relax_file), os.path.join(temp_relax_path, f"{count}"))

                if mode == 1:

                      energ_dft = dft_data[foldname][system_fold]

                      #data_energ.append((foldname, system_fold, energ_dft[0], energ_base, energ_relax, energ_dft[1]))
                      energ_dft_relax, energ_ref, n_at = float(energ_dft[0]), float(energ_dft[1]), int(energ_dft[2])

                      name_storage.append((foldname, system_fold, energ_dft_relax, energ_ref, n_at))

            elif mode == 2:
               

            count +=1


if force:
   command = rf"python .\run_alignn_ff.py --model_path {model_path} --file_path {temp_base_path} --task={task} --is_folder"
else:
  command = f"python .\pretrained.py --model_path {model_path} --file_format poscar --file_path {temp_base_path} --is_folder"
#command = f"python .\pretrained.py --model_name jv_formation_energy_peratom_alignn --file_format poscar --file_path {temp_base_path} --is_folder"

result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)

if multi:
  start_id = result.stdout.splitlines()[-1].index('[')


  base_data_list = ast.literal_eval(result.stdout.splitlines()[-1][start_id:])
  base_data_computed = [x[0] for x in base_data_list]

elif force:
   result.stdout = result.stdout.replace("\n", "").replace("\r", "")
   base_data_computed = [float(match.group(1)) for match in re.finditer(r"\((-?\d+\.\d+)", result.stdout)]

else:
  base_data_computed = extract_value(result.stdout.splitlines()[-1]).split(",")
#print(base_data_computed)

if force:
   command = rf"python .\run_alignn_ff.py --model_path {model_path} --file_path {temp_relax_path} --task={task} --is_folder"
else:
    command = f"python .\pretrained.py --model_path {model_path} --file_format poscar --file_path {temp_relax_path} --is_folder"
#command = f"python .\pretrained.py --model_name jv_formation_energy_peratom_alignn --file_format poscar --file_path {temp_relax_path} --is_folder"
result = subprocess.run(command, shell=True, capture_output=True, text=True)
#print(result)

if multi:
  start_id = result.stdout.splitlines()[-1].index('[')

  relax_data_list = ast.literal_eval(result.stdout.splitlines()[-1][start_id:])
  relax_data_computed = [x[0] for x in base_data_list]

elif force:
   result.stdout = result.stdout.replace("\n", "").replace("\r", "")
   relax_data_computed = [float(match.group(1)) for match in re.finditer(r"\((-?\d+\.\d+)", result.stdout)]

else:
  relax_data_computed = extract_value(result.stdout.splitlines()[-1]).split(",")


data_energ = []

#match the order of read files in POSCAR and the files in SUMMARY (DFT data)
match_index_list = [int(os.path.basename(file)) for file in glob.glob(temp_relax_path + "/*")]

for i in range(len(base_data_computed)):
    x = name_storage[match_index_list[i]]

    energ_dft_relax, energ_ref, n_at = x[2], x[3], x[4]

    energ_base = float(base_data_computed[i])
    energ_relax = float(relax_data_computed[i])

    #print(x, energ_base, energ_relax)

    if mean_energy_calc:
        energ_base = energ_base - energ_ref/n_at 
        energ_relax = energ_relax - energ_ref/n_at

    else:
        energ_base = (energ_base - energ_ref)/n_at 
        energ_relax = (energ_relax - energ_ref)/n_at
   
    #data_energ.append((x[0], x[1], (energ_dft_relax - energ_ref)/n_at, energ_base, energ_relax))
    data_energ.append((x[0], x[1], (energ_dft_relax - energ_ref)/n_at, energ_base, energ_relax))
save_to_drive(data_energ)

if os.path.exists(temp_base_path):
        shutil.rmtree(temp_base_path)

if os.path.exists(temp_relax_path):
        shutil.rmtree(temp_relax_path)
