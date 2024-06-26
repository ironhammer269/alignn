#!/usr/bin/env python

"""Module to download and load pre-trained ALIGNN models."""
import requests
import os
import zipfile
from tqdm import tqdm
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from data import get_torch_dataset
from torch.utils.data import DataLoader
import tempfile
import torch
import sys
import json
import glob

# from jarvis.db.jsonutils import loadjson
import argparse
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from jarvis.db.jsonutils import dumpjson
import pandas as pd

tqdm.pandas()

"""
Name of the model, figshare link, number of outputs,
extra config params (optional)
"""
# For ALIGNN-FF pretrained models see, alignn/ff/ff.py
all_models = {
    "jv_formation_energy_peratom_alignn": [
        "https://figshare.com/ndownloader/files/31458679",
        1,
    ],
    "jv_optb88vdw_total_energy_alignn": [
        "https://figshare.com/ndownloader/files/31459642",
        1,
    ],
    "jv_optb88vdw_bandgap_alignn": [
        "https://figshare.com/ndownloader/files/31459636",
        1,
    ],
    "jv_mbj_bandgap_alignn": [
        "https://figshare.com/ndownloader/files/31458694",
        1,
    ],
    "jv_spillage_alignn": [
        "https://figshare.com/ndownloader/files/31458736",
        1,
    ],
    "jv_slme_alignn": ["https://figshare.com/ndownloader/files/31458727", 1],
    "jv_bulk_modulus_kv_alignn": [
        "https://figshare.com/ndownloader/files/31458649",
        1,
    ],
    "jv_shear_modulus_gv_alignn": [
        "https://figshare.com/ndownloader/files/31458724",
        1,
    ],
    "jv_n-Seebeck_alignn": [
        "https://figshare.com/ndownloader/files/31458718",
        1,
    ],
    "jv_n-powerfact_alignn": [
        "https://figshare.com/ndownloader/files/31458712",
        1,
    ],
    "jv_magmom_oszicar_alignn": [
        "https://figshare.com/ndownloader/files/31458685",
        1,
    ],
    "jv_kpoint_length_unit_alignn": [
        "https://figshare.com/ndownloader/files/31458682",
        1,
    ],
    "jv_avg_elec_mass_alignn": [
        "https://figshare.com/ndownloader/files/31458643",
        1,
    ],
    "jv_avg_hole_mass_alignn": [
        "https://figshare.com/ndownloader/files/31458646",
        1,
    ],
    "jv_epsx_alignn": ["https://figshare.com/ndownloader/files/31458667", 1],
    "jv_mepsx_alignn": ["https://figshare.com/ndownloader/files/31458703", 1],
    "jv_max_efg_alignn": [
        "https://figshare.com/ndownloader/files/31458691",
        1,
    ],
    "jv_ehull_alignn": ["https://figshare.com/ndownloader/files/31458658", 1],
    "jv_dfpt_piezo_max_dielectric_alignn": [
        "https://figshare.com/ndownloader/files/31458652",
        1,
    ],
    "jv_dfpt_piezo_max_dij_alignn": [
        "https://figshare.com/ndownloader/files/31458655",
        1,
    ],
    "jv_exfoliation_energy_alignn": [
        "https://figshare.com/ndownloader/files/31458676",
        1,
    ],
    "jv_supercon_tc_alignn": [
        "https://figshare.com/ndownloader/files/38789199",
        1,
    ],
    "jv_supercon_edos_alignn": [
        "https://figshare.com/ndownloader/files/39946300",
        1,
    ],
    "jv_supercon_debye_alignn": [
        "https://figshare.com/ndownloader/files/39946297",
        1,
    ],
    "jv_supercon_a2F_alignn": [
        "https://figshare.com/ndownloader/files/38801886",
        100,
    ],
    "mp_e_form_alignn": [
        "https://figshare.com/ndownloader/files/31458811",
        1,
    ],
    "mp_gappbe_alignn": [
        "https://figshare.com/ndownloader/files/31458814",
        1,
    ],
    "tinnet_O_alignn": ["https://figshare.com/ndownloader/files/41962800", 1],
    "tinnet_N_alignn": ["https://figshare.com/ndownloader/files/41962797", 1],
    "tinnet_OH_alignn": ["https://figshare.com/ndownloader/files/41962803", 1],
    "AGRA_O_alignn": ["https://figshare.com/ndownloader/files/41966619", 1],
    "AGRA_OH_alignn": ["https://figshare.com/ndownloader/files/41966610", 1],
    "AGRA_CHO_alignn": ["https://figshare.com/ndownloader/files/41966643", 1],
    "AGRA_CO_alignn": ["https://figshare.com/ndownloader/files/41966634", 1],
    "AGRA_COOH_alignn": ["https://figshare.com/ndownloader/41966646", 1],
    "qm9_U0_alignn": ["https://figshare.com/ndownloader/files/31459054", 1],
    "qm9_U_alignn": ["https://figshare.com/ndownloader/files/31459051", 1],
    "qm9_alpha_alignn": ["https://figshare.com/ndownloader/files/31459027", 1],
    "qm9_gap_alignn": ["https://figshare.com/ndownloader/files/31459036", 1],
    "qm9_G_alignn": ["https://figshare.com/ndownloader/files/31459033", 1],
    "qm9_HOMO_alignn": ["https://figshare.com/ndownloader/files/31459042", 1],
    "qm9_LUMO_alignn": ["https://figshare.com/ndownloader/files/31459045", 1],
    "qm9_ZPVE_alignn": ["https://figshare.com/ndownloader/files/31459057", 1],
    "hmof_co2_absp_alignn": [
        "https://figshare.com/ndownloader/files/31459198",
        5,
    ],
    "hmof_max_co2_adsp_alignn": [
        "https://figshare.com/ndownloader/files/31459207",
        1,
    ],
    "hmof_surface_area_m2g_alignn": [
        "https://figshare.com/ndownloader/files/31459222",
        1,
    ],
    "hmof_surface_area_m2cm3_alignn": [
        "https://figshare.com/ndownloader/files/31459219",
        1,
    ],
    "hmof_pld_alignn": ["https://figshare.com/ndownloader/files/31459216", 1],
    "hmof_lcd_alignn": ["https://figshare.com/ndownloader/files/31459201", 1],
    "hmof_void_fraction_alignn": [
        "https://figshare.com/ndownloader/files/31459228",
        1,
    ],
    "ocp2020_all": ["https://figshare.com/ndownloader/files/41411025", 1],
    "ocp2020_100k": ["https://figshare.com/ndownloader/files/41967303", 1],
    "ocp2020_10k": ["https://figshare.com/ndownloader/files/41967330", 1],
    "jv_pdos_alignn": [
        "https://figshare.com/ndownloader/files/36757005",
        66,
        {"alignn_layers": 6, "gcn_layers": 6},
    ],
}


parser = argparse.ArgumentParser(
    description="Atomistic Line Graph Neural Network Pretrained Models"
)
parser.add_argument(
    "--model_name",
    default="jv_formation_energy_peratom_alignn",
    help="Choose a model from these "
    + str(len(list(all_models.keys())))
    + " models:"
    + ", ".join(list(all_models.keys())),
)

parser.add_argument(
    "--model_path",
    default="NA",
    help="Model path for use of a locally saved model"
)

parser.add_argument(
    "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
)

parser.add_argument(
    "--is_folder",  action='store_true', help="Set to true if path is a folder"
)

parser.add_argument(
    "--use_ff",  action='store_true', help="Set to true if model uses forces"
)

parser.add_argument(
    "--file_path",
    default="alignn/examples/sample_data/POSCAR-JVASP-10.vasp",
    help="Path to file.",
)

parser.add_argument(
    "--cutoff",
    default=8,
    help="Distance cut-off for graph constuction"
    + ", usually 8 for solids and 5 for molecules.",
)

parser.add_argument(
    "--max_neighbors",
    default=12,
    help="Maximum number of nearest neighbors in the periodic atomistic graph"
    + " construction.",
)

parser.add_argument(
    "--batch_size",
    default=8,
    help="Batch size for loader.",
)


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

# device = "cpu"


def get_all_models():
    """Return the figshare links for models."""
    return all_models


def get_figshare_model(model_name="jv_formation_energy_peratom_alignn"):
    """Get ALIGNN torch models from figshare."""
    # https://figshare.com/projects/ALIGNN_models/126478

    tmp = all_models[model_name]
    url = tmp[0]
    # output_features = tmp[1]
    # if len(tmp) > 2:
    #    config_params = tmp[2]
    # else:
    #    config_params = {}
    zfile = model_name + ".zip"
    path = str(os.path.join(os.path.dirname(__file__), zfile))
    if not os.path.isfile(path):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes, unit="iB", unit_scale=True
        )
        with open(path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    zp = zipfile.ZipFile(path)
    names = zp.namelist()
    chks = []
    cfg = []
    for i in names:
        if "checkpoint_" in i and "pt" in i:
            tmp = i
            chks.append(i)
        if "config.json" in i:
            cfg = i

    print("Using chk file", tmp, "from ", chks)
    print("Path", os.path.abspath(path))
    print("Config", os.path.abspath(cfg))
    config = json.loads(zipfile.ZipFile(path).read(cfg))
    # print("Loading the zipfile...", zipfile.ZipFile(path).namelist())
    data = zipfile.ZipFile(path).read(tmp)
    # model = ALIGNN(
    #    ALIGNNConfig(
    #        name="alignn", output_features=output_features, **config_params
    #    )
    # )
    model = ALIGNN(ALIGNNConfig(**config["model"]))

    new_file, filename = tempfile.mkstemp()
    with open(filename, "wb") as f:
        f.write(data)
    model.load_state_dict(torch.load(filename, map_location=device)["model"])
    model.to(device)
    model.eval()
    os.close(new_file)
    if os.path.exists(filename):
        os.remove(filename)
    return model

def get_local_model(path):
    tmp = os.path.join(path, "best_model.pt")
    cfg = os.path.join(path, "config.json")
    print("Using chk file", tmp)
    print("Path", os.path.abspath(path))
    print("Config", os.path.abspath(cfg))
    with open(cfg, 'r') as f:
        config = json.load(f)

    #model = ALIGNN(ALIGNNConfig(**config["model"]))
    model = ALIGNNAtomWise(ALIGNNAtomWiseConfig(**config["model"]))
    model.load_state_dict(torch.load(tmp, map_location=device))
    model.to(device)
    model.eval()

    print(model)

    return model

def get_prediction(
    model_name="jv_formation_energy_peratom_alignn",
    model_path="NA",
    atoms=None,
    cutoff=8,
    max_neighbors=12,
):
    """Get model prediction on a single structure."""
    if model_path == "NA":
        model = get_figshare_model(model_name)
    else:
        model = get_local_model(model_path)
    # print("Loading completed.")
    g, lg = Graph.atom_dgl_multigraph(
        atoms,
        cutoff=float(cutoff),
        max_neighbors=max_neighbors,
        use_canonize=True
    )

    out_data = (
        model([g.to(device), lg.to(device)])['out']
        #model([g.to(device), lg.to(device)])
        .detach()
        .cpu()
        .numpy()
        .flatten()
        .tolist()
    )
    return out_data


def get_multiple_predictions(
    atoms_array=[],
    cutoff=8,
    neighbor_strategy="k-nearest",
    max_neighbors=12,
    use_canonize=True,
    target="prop",
    atom_features="cgcnn",
    line_graph=True,
    workers=0,
    filename="pred_data.json",
    include_atoms=True,
    pin_memory=False,
    output_features=1,
    batch_size=1,
    model=None,
    model_name="jv_formation_energy_peratom_alignn",
    model_path="NA",
    use_ff=False,
    print_freq=100,
):
    """Use pretrained model on a number of structures."""

    mem = []
    for i, ii in enumerate(atoms_array):
        info = {}
        info["atoms"] = ii.to_dict()
        info["prop"] = -9999  # place-holder only
        info["jid"] = str(i)
        mem.append(info)

    if model is None:
        try:
            if model_path == "NA":
                model = get_figshare_model(model_name)
            else:
                model = get_local_model(model_path)
        except Exception as exp:
            raise ValueError(
                'Check is the model name exists using "pretrained.py -h"', exp
            )
            pass

    
    test_data = get_torch_dataset(
        dataset=mem,
        target="prop",
        neighbor_strategy=neighbor_strategy,
        atom_features=atom_features,
        use_canonize=use_canonize,
        line_graph=line_graph,
    )

    collate_fn = test_data.collate_line_graph
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    results = []
    with torch.no_grad():
        ids = test_loader.dataset.ids
        for dat, id in zip(test_loader, ids):
            g, lg, target = dat

            if model_path == "NA":
                out_data = model([g.to(device), lg.to(device)])
            else : 
                out_data = model([g.to(device), lg.to(device)])['out']
            out_data = out_data.cpu().numpy().tolist()
            target = target.cpu().numpy().flatten().tolist()
            results += out_data
            print_freq = int(print_freq)
            if len(results) % print_freq == 0:
                print(len(results))


                
    return results


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    model_name = args.model_name
    model_path = args.model_path
    use_ff = args.use_ff
    file_path = args.file_path
    file_format = args.file_format
    is_folder = args.is_folder
    cutoff = args.cutoff
    max_neighbors = args.max_neighbors
    batch_size = args.batch_size
    
    if is_folder:
        atoms_array = []
        out_data = []
        for i in glob.glob(file_path + "/*"):
            
            if file_format == "poscar":
                try :
                    atoms = Atoms.from_poscar(i)
                except :
                    print("Error in file:", i)
            elif file_format == "cif":
                atoms = Atoms.from_cif(i)
            elif file_format == "xyz":
                atoms = Atoms.from_xyz(i, box_size=500)
            elif file_format == "pdb":
                atoms = Atoms.from_pdb(i, max_lat=500)
            else:
                raise NotImplementedError("File format not implemented", file_format)
                
            atoms_array.append(atoms)
        
        if len(atoms_array)>0:
            out_data = get_multiple_predictions(
                model_name=model_name,
                model_path=model_path,
                use_ff=use_ff,
                cutoff=float(cutoff),
                max_neighbors=int(max_neighbors),
                atoms_array=atoms_array,
                batch_size = int(batch_size),
            )
        
    else: 
        if file_format == "poscar":
            atoms = Atoms.from_poscar(file_path)
        elif file_format == "cif":
            atoms = Atoms.from_cif(file_path)
        elif file_format == "xyz":
            atoms = Atoms.from_xyz(file_path, box_size=500)
        elif file_format == "pdb":
            atoms = Atoms.from_pdb(file_path, max_lat=500)
        else:
            raise NotImplementedError("File format not implemented", file_format)
    
        out_data = get_prediction(
            model_name=model_name,
            model_path=model_path,
            cutoff=float(cutoff),
            max_neighbors=int(max_neighbors),
            atoms=atoms,
        )

    print("Predicted value:", file_path, out_data)
    # import glob
    # atoms_array = []
    # for i in glob.glob("alignn/examples/sample_data/*.vasp"):
    #    atoms = Atoms.from_poscar(i)
    #    atoms_array.append(atoms)
    # get_multiple_predictions(atoms_array=atoms_array)
