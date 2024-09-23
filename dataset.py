import json
import codecs
import random
import numpy as np
import networkx as nx
import torch
from rdkit import Chem
from torch_geometric.data import InMemoryDataset, Dataset
from torch_geometric.utils.convert import from_networkx
from typing import List, Union
from torch_geometric.data.data import Data

# --- Class variables
ATOM_FEATURES = {
    'atomic_num': [6.0, 8.0],
    'degree': [1, 2, 3, 4],
    'num_Hs': [0.0, 1.0, 2.0],
    'num_Os': [0.0, 1.0, 2.0],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3
    ]
}

# --- Total number of atom features
ATOM_FDIM = sum(len(choices) for choices in ATOM_FEATURES.values()) + 1
# --- Number of bond features
BOND_FDIM = 14

def mol_to_nx(mol, spec, atom_rep=True):
    '''
    text
    '''

    # --- Create graph object
    G = nx.Graph()
        
    # --- For each atom in molecule
    for atom in mol.GetAtoms():
        # --- Add a node to graph and create one-hot encoding vector for atom features
        map_num = atom.GetAtomMapNum()
        G.add_node(map_num, x=get_atom_features(atom))
            
    # --- For each bond in molecule
    for bond in mol.GetBonds():
        # ---
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        begin_map = begin.GetAtomMapNum()
        end_map = end.GetAtomMapNum()
        # --- Add edge to graph and create one-hot encoding vector of bond features
        G.add_edge(begin_map, end_map, edge_attr=get_bond_features(bond))

    if atom_rep == False:
        # --- Normalize spectra to 1.0
        max_intensity = np.max(spec)
        norm_spec = 1.0 * (spec / max_intensity)
        # --- Set spectra to graph
        G.graph['spectrum'] = torch.FloatTensor(norm_spec)
        
    return G
    
def get_atom_features(atom) -> List[Union[bool, int, float]]:
    '''
    Builds a feature vector for an atom

    :param atom: An RDKit atom
    :return: A list containing the atom features
    '''

    # --- For one-hot encoding featue vector
    num_Os = 0
    if atom is None:
        features = [0] * ATOM_FDIM
    else:
        for a in atom.GetNeighbors():
            if a.GetAtomicNum() == 8:
                num_Os += 1.0
        # features = one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']) + \
        #     [atom.GetDegree()] + [atom.GetTotalNumHs()] + [num_Os] + \
        #     one_hot_encoding(atom.GetHybridization(), ATOM_FEATURES['hybridization']) + \
        #     [1 if atom.GetIsAromatic() else 0]
        # --- Get the values of all the atom features and add all up to the feature vector
        features = one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']) + \
            one_hot_encoding(atom.GetDegree(), ATOM_FEATURES['degree']) + \
            one_hot_encoding(atom.GetTotalNumHs(), ATOM_FEATURES['num_Hs']) + \
            one_hot_encoding(num_Os, ATOM_FEATURES['num_Os']) + \
            one_hot_encoding(atom.GetHybridization(), ATOM_FEATURES['hybridization']) + \
            [1 if atom.GetIsAromatic() else 0]

    return features        

def get_bond_features(bond) -> List[Union[bool, int, float]]:
    '''
    Builds a features vector for a bond

    :params bond: An RDKit bond
    :return: A list containing the bond features
    '''

    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        # --- Creat one-hot bond vector
        fbond = [
            int(bt == Chem.rdchem.BondType.SINGLE),
            int(bt == Chem.rdchem.BondType.DOUBLE),
            int(bt == Chem.rdchem.BondType.AROMATIC),
            int(bond.GetIsConjugated() if bt is not None else 0),
            int(bond.IsInRing() if bt is not None else 0)
        ]


    return fbond

def one_hot_encoding(value:int, choices:List[int]) -> List[int]:
    '''
    Creates a one-hot encoding
    '''

    # --- Create a zero atom feature vector
    encoding = [0.0] * len(choices)
    # --- Find the index value of
    index = choices.index(value)
    # --- Set value to 1
    encoding[index] = 1.0

    return encoding

def count_atoms(mol, atomic_num):
    '''
    text
    '''

    # --- 
    num_atoms = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == atomic_num:
            num_atoms += 1
    return num_atoms


class XASDataset_mol(InMemoryDataset):
    '''
    Text
    '''

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
 
    @property
    def raw_file_names(self):
        # return ['data_coronene.json']
        return ['data_circumcoronene.json']

    @property
    def processed_file_names(self):
        return ['circ_xasnet.pt']
    
    def process(self):
        '''
        Text
        '''

        # --- List to store data
        data_list = []

        # --- Load the raw data from json file
        dat = codecs.open(self.raw_paths[0], 'r', encoding='utf-8')
        dictionaires = json.load(dat)

        # --- Create list with all the molecule names
        all_names = list(dictionaires[0].keys())
        print(f'Total number of molecules {len(all_names)}')

        # --- 
        idx = 0
        
        for name in all_names:
            # --- 
            smiles = dictionaires[0][name]
            mol = Chem.MolFromSmiles(smiles)
            # ---
            atom_spec = dictionaires[1][name]
 
            # # --- Create arrays of dataset
            # pos = dictionaires[0][name][1]
            # positions = np.array(pos)
            # z_num = dictionaires[0][name][2]
            # z = np.array(z_num)
            # atom_count = count_atoms(mol, 6)

            tot_spec = np.zeros(len(atom_spec[str(0)]))

            for key in atom_spec.keys():
                # --- Sum up all atomic spectra
                tot_spec += atom_spec[key]

            # --- Create graph objects
            # --- For schnet models
            # dict = {}
            # dict['x'] = None
            # pyg_graph = Data.from_dict(dict)
            # pyg_graph.pos = torch.from_numpy(positions)
            # pyg_graph.z = torch.from_numpy(z)

            # --- For PyG models
            gx = mol_to_nx(mol, tot_spec, atom_rep=False)
            pyg_graph = from_networkx(gx)

            pyg_graph.idx = idx
            pyg_graph.smiles = smiles
            data_list.append(pyg_graph)
            idx += 1

        random.Random(258).shuffle(data_list)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
