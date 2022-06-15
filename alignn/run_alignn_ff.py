#!/usr/bin/env python

"""Module to run ALIGNN-FF."""
import argparse
import sys
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from alignn.ff.ff import (
    default_path,
    ev_curve,
    surface_energy,
    vacancy_formation,
    ForceField,
)


parser = argparse.ArgumentParser(
    description="Atomistic Line Graph Neural Network Force-field"
)
parser.add_argument(
    "--model_path",
    default="na",
    help="Provide model path, id na, default path chosen ",
)

parser.add_argument(
    "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
)

parser.add_argument("--on_relaxed_struct", default="No", help="Yes/No.")
parser.add_argument(
    "--file_path",
    default="alignn/examples/sample_data/POSCAR-JVASP-10.vasp",
    help="Path to file.",
)

parser.add_argument(
    "--task",
    default="optimize",
    help="Select task for ALIGNN-FF"
    + " such as optimze/nvt_lagevin/nve_velocity_verlet/"
    + "ev_curve/vacancy_energy/surface_energy",
)

parser.add_argument("--md_steps", default=100, help="Provide md steps.")


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    model_path = args.model_path
    file_path = args.file_path
    file_format = args.file_format
    task = args.task
    on_relaxed_struct_yn = args.on_relaxed_struct
    if on_relaxed_struct_yn.lower() == "yes":
        on_relaxed_struct = True
    else:
        on_relaxed_struct = False

    steps = int(args.md_steps)
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
    if model_path == "na":
        model_path = default_path()
    if task == "optimize":
        ff = ForceField(
            jarvis_atoms=atoms,
            model_path=model_path,
        )
        opt = ff.optimize_atoms()
        print("initial struct:")
        print(atoms)
        print("final struct:")
        print(opt)
    if task == "nvt_lagevin":
        print("initial struct:")
        print(atoms)
        ff = ForceField(
            jarvis_atoms=atoms,
            model_path=model_path,
        )
        lang = ff.run_nvt_langevin(steps=steps)
        print("final struct:")
        print(lang)

    if task == "nve_velocity_verlet":
        print("initial struct:")
        print(atoms)
        ff = ForceField(
            jarvis_atoms=atoms,
            model_path=model_path,
        )
        vv = ff.run_nve_velocity_verlet(steps=steps)
        print("final struct:")
        print(vv)
    if task == "ev_curve":
        ff = ForceField(
            jarvis_atoms=atoms,
            model_path=model_path,
        )
        ev = ev_curve(
            atoms=atoms,
            model_path=model_path,
            on_relaxed_struct=on_relaxed_struct,
        )
    if task == "vacancy_energy":
        ff = ForceField(
            jarvis_atoms=atoms,
            model_path=model_path,
        )
        vac = vacancy_formation(
            atoms=atoms,
            model_path=model_path,
            on_relaxed_struct=on_relaxed_struct,
        )
        print(vac)
    if task == "surface_energy":
        ff = ForceField(
            jarvis_atoms=atoms,
            model_path=model_path,
        )
        surf = surface_formation(
            atoms=atoms,
            model_path=model_path,
            on_relaxed_struct=on_relaxed_struct,
        )
        print(surf)