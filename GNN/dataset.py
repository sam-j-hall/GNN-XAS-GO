import json
import codecs
import pickle as pkl
import numpy as np
import networkx as nx
import torch
from rdkit import Chem
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils.convert import from_networkx
from typing import List, Union

# Node features
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

BOND_FEATURES = {
    'bond_type' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.AROMATIC
    ]
}

def mol_to_nx(mol):
    '''
    Creates a PyG graph object with corresponding spectra
    from a RDKit molecule

    :param mol: An RDKit molecule
    :param spec: A numpy array of the XAS
    :return G: A PyG graph object
    '''

    # Create graph object
    G = nx.Graph()
        
    # For each atom in molecule
    for atom in mol.GetAtoms():
        # Add a node to graph and create one-hot encoding vector for atom features
        atom_num = atom.GetIdx()
        G.add_node(atom_num, x=get_atom_features(atom))
            
    # For each bond in molecule
    for bond in mol.GetBonds():
        # Get atoms numbers of bond
        begin = bond.GetBeginAtom().GetIdx()
        end = bond.GetEndAtom().GetIdx()
        # Add edge to graph and create one-hot encoding vector of bond features
        G.add_edge(begin, end, edge_attr=get_bond_features(bond))
        
    return G
    
def get_atom_features(atom) -> List[Union[bool, int, float]]:
    '''
    Builds a feature vector for an atom

    :param atom: An RDKit atom
    :return: A list containing the atom features
    '''

    # For one-hot encoding featue vector
    # Get the values of all the atom features and add all up to the feature vector
    atom_feat = one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']) + \
        one_hot_encoding(atom.GetDegree(), ATOM_FEATURES['degree']) + \
        one_hot_encoding(atom.GetTotalNumHs(), ATOM_FEATURES['num_Hs']) + \
        one_hot_encoding(atom.GetHybridization(), ATOM_FEATURES['hybridization']) + \
        [1.0 if atom.GetIsAromatic() else 0.0]

    return atom_feat       

def get_bond_features(bond) -> List[Union[bool, int, float]]:
    '''
    Builds a features vector for a bond

    :params bond: An RDKit bond
    :return: A list containing the bond features
    '''

    # Create one-hot bond vector
    bond_feat = one_hot_encoding(bond.GetBondType(), BOND_FEATURES['bond_type']) + \
        [1.0 if bond.GetIsConjugated() else 0.0] + \
        [1.0 if bond.IsInRing() else 0.0]
    
    return bond_feat

def one_hot_encoding(value:int, choices:List[int]) -> List[int]:
    '''
    Creates a one-hot encoding
    '''

    # Create a zero atom feature vector
    encoding = [0.0] * len(choices)
    # Find the index value of
    index = choices.index(value)
    #  Set value to 1
    encoding[index] = 1.0

    return encoding


class XASMolDataset(InMemoryDataset):
    '''
    Text
    '''

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
 
    @property
    def raw_file_names(self):
        return ['data_coronene_voigt.pkl']
        # return ['data_circumcoronene.json']

    @property
    def processed_file_names(self):
        return ['coronene_pyg_voigt.pt']
        # return ['circumcoronene_pyg.pt']
    
    def process(self):
        '''
        Text
        '''

        # List to store data
        data_list = []

        # Load the raw data from pickle file
        dat_file = open(self.raw_paths[0], 'rb')
        dat_df = pkl.load(dat_file)
        # dat = codecs.open(self.raw_paths[0], 'r', encoding='utf-8')
        # dictionaires = json.load(dat)

        print(f'Total number of molecules {len(dat_df)}')

        # --- 
        idx = 0
        
        for index, row in dat_df.iterrows():
            # Read RDKit mol from smiles
            smiles = row['SMILES']
            mol = Chem.MolFromSmiles(smiles)
            # Get spectrum
            spec = row['Spectrum']

            gx = mol_to_nx(mol)
            pyg_graph = from_networkx(gx)

            # Normalize spectra to 1.0
            max_intensity = np.max(spec)
            norm_spec = 1.0 * (spec / max_intensity)
            # Set spectra to graph
            pyg_graph.spectrum = torch.FloatTensor(norm_spec)

            pyg_graph.idx = idx
            pyg_graph.smiles = smiles
            data_list.append(pyg_graph)
            idx += 1

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class SchNetDataset(InMemoryDataset):
    '''
    Text
    '''

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
 
    @property
    def raw_file_names(self):
        return ['data_coronene.json']
        # return ['data_circumcoronene.json']

    @property
    def processed_file_names(self):
        return ['coronene_schnet.pt']
    
    def process(self):
        '''
        Text
        '''

        # List to store data
        data_list = []

        # Load the raw data from json file
        dat = codecs.open(self.raw_paths[0], 'r', encoding='utf-8')
        dictionaires = json.load(dat)

        # Create list with all the molecule names
        all_names = list(dictionaires[0].keys())
        print(f'Total number of molecules {len(all_names)}')

        # --- 
        idx = 0
        
        for name in all_names:
            # Get pos and z data
            pos = torch.Tensor(dictionaires[2][name][0])
            z = torch.Tensor(dictionaires[2][name][1]).long()

            smiles = dictionaires[0][name]
            mol = Chem.MolFromSmiles(smiles)
            atom_spec = dictionaires[1][name]

            tot_spec = np.zeros(len(atom_spec[str(0)]))
            for key in atom_spec.keys():
                tot_spec += atom_spec[key]

            # Normalise spectrum to 1
            max_int = np.max(tot_spec)
            norm_spec = 1.0 * (tot_spec / max_int)
            norm_spec = np.float32(norm_spec)

            pyg_graph = Data(x=pos, z=z)
            pyg_graph.spectrum = torch.Tensor(norm_spec)
            pyg_graph.idx = idx
            pyg_graph.smiles = smiles

            data_list.append(pyg_graph)
            idx += 1

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class XASAtomDataset(InMemoryDataset):
    '''
    Text
    '''

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
 
    @property
    def raw_file_names(self):
        return ['data_coronene.json']
        # return ['data_circumcoronene.json']

    @property
    def processed_file_names(self):
        return ['coronene_pyg_atom.pt']
        # return ['circumcoronene_pyg.pt']
    
    def process(self):
        '''
        Text
        '''

        # List to store data
        data_list = []

        # Load the raw data from json file
        dat = codecs.open(self.raw_paths[0], 'r', encoding='utf-8')
        dictionaires = json.load(dat)

        # Create list with all the molecule names
        all_names = list(dictionaires[0].keys())
        print(f'Total number of molecules {len(all_names)}')

        # --- 
        idx = 0
        
        for name in all_names:
            # 
            smiles = dictionaires[0][name]
            mol = Chem.MolFromSmiles(smiles)
            # 
            atom_spec = dictionaires[1][name]

            atom_count = 0
            for atom in mol.GetAtoms():
                atom_count += 1

            spec_list = torch.zeros(atom_count, 200)

            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 6:
                    sp = atom_spec[str(atom.GetIdx())]
                    spec_list[atom.GetIdx()] = torch.DoubleTensor(sp)

            gx = mol_to_nx(mol)
            pyg_graph = from_networkx(gx)
            pyg_graph.spectrum = spec_list
            pyg_graph.idx = idx
            pyg_graph.smiles = smiles
            pyg_graph.mol = name
            data_list.append(pyg_graph)
            idx += 1


            # #
            # for atom in mol.GetAtoms():
            #     if atom.GetAtomicNum() == 6:
            #         spec = atom_spec[str(atom.GetIdx())]


            #     gx = mol_to_nx(mol)
            #     pyg_graph = from_networkx(gx)
            #     pyg_graph.spectrum = torch.FloatTensor(spec)
            #     pyg_graph.atom_num = atom.GetIdx()
            #     pyg_graph.idx = idx
            #     pyg_graph.smiles = smiles
            #     pyg_graph.mol = name
            #     data_list.append(pyg_graph)
            #     idx += 1

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])