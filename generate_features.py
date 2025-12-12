#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate molecular features (Fingerprints and Graph features) for GAT-GCN model.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import pickle

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    """
    Generate atom features (Node features).
    Dimension: 44 (approx, depending on implementation)
    """
    # Atom type (Symbol)
    results = one_of_k_encoding(atom.GetSymbol(), 
                                ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])
    
    # Degree
    results += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # Implicit Valence
    results += one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
    
    # Formal Charge
    results += [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
    
    # Hybridization
    results += one_of_k_encoding(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, 'other'
    ])
    
    # Aromaticity
    results += [atom.GetIsAromatic()]
    
    # Total Num Hs
    results += one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    
    return np.array(results).astype(float)

def bond_features(bond):
    """
    Generate bond features (Edge features).
    """
    bt = bond.GetBondType()
    results = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return np.array(results).astype(float)

def generate_fingerprint(mol, nBits=1024, radius=2):
    """Generate ECFP4 fingerprint"""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp)

def process_smiles(smiles):
    """Process a single SMILES string to get graph features and fingerprint"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    
    # Node features
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append(atom_features(atom))
    atom_feats = np.array(atom_feats)
    
    # Edge features and Adjacency
    # Note: For GCN, we usually need Adjacency matrix or Edge List + Edge Attr
    # Here we return Edge List and Edge Attr
    edges = []
    edge_feats = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Add both directions
        edges.append([i, j])
        edge_feats.append(bond_features(bond))
        
        edges.append([j, i])
        edge_feats.append(bond_features(bond))
        
    if not edges: # Single atom or no bonds
        edges = np.zeros((0, 2), dtype=int)
        edge_feats = np.zeros((0, 6), dtype=float)
    else:
        edges = np.array(edges)
        edge_feats = np.array(edge_feats)
        
    # Fingerprint
    fp = generate_fingerprint(mol)
    
    return atom_feats, (edges, edge_feats), fp

def main(input_path=None):
    if input_path:
        input_file = Path(input_path)
    else:
        input_file = Path("./data/raw/vegfr2_processed.csv")
    output_dir = Path("./data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_file}")
        return
        
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Check required columns
    if 'canonical_smiles' not in df.columns:
        print("[ERROR] 'canonical_smiles' column missing.")
        return
        
    print(f"Processing {len(df)} compounds...")
    
    data_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row['canonical_smiles']
        labels = {}
        if 'label_vegfr2' in row:
            labels['vegfr2'] = row['label_vegfr2']
        if 'label_a549' in row:
            labels['a549'] = row['label_a549']
            
        atom_feats, graph_feats, fp = process_smiles(smiles)
        
        if atom_feats is None:
            continue
            
        data_list.append({
            'smiles': smiles,
            'atom_features': atom_feats,
            'edge_index': graph_feats[0],
            'edge_features': graph_feats[1],
            'fingerprint': fp,
            'labels': labels,
            'chembl_id': row.get('molecule_chembl_id', '')
        })
        
    print(f"Successfully processed {len(data_list)} compounds.")
    
    # Save as pickle
    output_pkl = output_dir / "features_data.pkl"
    with open(output_pkl, 'wb') as f:
        pickle.dump(data_list, f)
        
    print(f"[OK] Features saved to {output_pkl}")
    
    # Also save fingerprints separately as npy if needed
    fps = np.array([d['fingerprint'] for d in data_list])
    np.save(output_dir / "fingerprints.npy", fps)
    print(f"[OK] Fingerprints saved to {output_dir / 'fingerprints.npy'}")

if __name__ == "__main__":
    main()
