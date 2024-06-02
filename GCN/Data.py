import cv2
import torch
import numpy as np
import pandas as pd
from skimage import measure
import networkx as nx
import matplotlib.pyplot as plt
import imageio
import torch
from torch_geometric.data import Data
import networkx as nx
from scipy.spatial import Delaunay
#from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Dataset, DataLoader


class graphDS(Dataset):
    def __init__(self,test, test_patient, m, target = 'Recurrence'):
        self.target = target
        self.morphology_model = m
        if test == True:
            ## path to cellsData
            self.protein = pd.read_csv(r"cellData.csv") 
            self.protein = self.protein[self.protein['SampleID'] == test_patient]
        else:
            self.protein = pd.read_csv(r"../../MIBI-TNBC/cellData.csv")
            self.protein = self.protein[(self.protein['SampleID'] != test_patient) & (self.protein['SampleID'] < 41)]
        ### path to clinic annot
        meta_data = pd.read_excel(r"../../MIBI-TNBC/mmc2.xlsx", index_col= 0).iloc[1:-3,:].dropna(axis = 1) 
        #le = preprocessing.LabelEncoder()
        mapper = {'Unnamed: 1': 'DONOR_NO', 'Unnamed: 2': 'YEAR', 'Unnamed: 3':'ANON_ID', 'Unnamed: 4': 'AGE_AT_DX',
                'Unnamed: 5':'YEAR_DX', 'Unnamed: 6':'STAGE',
                'Unnamed: 7': 'SITE_02', 'Unnamed: 8':'LATERAL', 'Unnamed: 9':'GRADE', 'Unnamed: 14':'ER',
                'Unnamed: 15':'PR', 'Unnamed: 16':'HER2NEU','Unnamed: 25':'Recorrence', 'Unnamed: 27':'Survival_days',
                'Unnamed: 28':'Censored'}
        meta_data.rename(columns = mapper, inplace = True)
        irrelevant_ft = ['DONOR_NO', 'YEAR', 'ANON_ID', 'YEAR_DX', 'ER', 'PR', 'HER2NEU']
        meta_data.drop(irrelevant_ft, axis = 1, inplace = True)
        meta_df = meta_data.apply(pd.to_numeric, errors='ignore')
        meta_df['LATERAL'] = meta_df['LATERAL'] - 1
        self.meta = meta_df
        self.meta['Recorrence'] = self.meta['Recorrence'].map({'POSITIVE':1, 'NEGATIVE':0})
        self.meta['GRADE'] = self.meta['GRADE'].map({1:0,2:1,3:2,4:2,9:2})
        
        ### props list
        self.props_list = ['label', 'centroid', 'area', 'eccentricity','major_axis_length',
                           'minor_axis_length','perimeter','equivalent_diameter_area','area_convex','extent',
                           'feret_diameter_max','orientation','perimeter_crofton','solidity']
    
    def __len__(self):
        return len(np.unique(self.protein['SampleID']))
    def __getitem__(self, index):
        index = np.unique(self.protein['SampleID'])[index]
        ### path to labeled-image
        labels = imageio.imread(fr"Point{index}/p{index}_labeledcellData.tiff").astype(np.int16)
        props_single_cell = pd.DataFrame(measure.regionprops_table(labels, properties = self.props_list))
        self.patient_protein = self.protein[self.protein['SampleID'] == index].drop(['Background', 'B7H3', 'OX40', 'CD163', 'CSF-1R']
                                                                                    , axis = 1)
        ### merge
        self.morph_props = pd.merge(self.patient_protein, props_single_cell, how = 'inner', left_on = ['cellLabelInImage'],
                                  right_on = ['label']).drop(['label', 'Ta', 'Au', 'tumorYN',
                                                               'tumorCluster', 'Group', 'immuneCluster', 'immuneGroup',
                                                              
                                                             ], axis = 1)
        
        graph = {'ft_radios':[], 'ft_index' : [], 'ft_coords' : []}
        coords_Y = self.morph_props['centroid-0'].tolist()
        coords_X = self.morph_props['centroid-1'].tolist()
        coords = [[i,j] for i,j in zip(coords_X, coords_Y)]
        points = np.array(coords)
        indptr_neigh, neighbours = Delaunay(points).vertex_neighbor_vertices
        edge = []
        self.node_ft = []
        for i,cell_label in enumerate(self.morph_props['cellLabelInImage'].tolist()):
            if self.morphology_model == True:
                pp = self.morph_props.drop(['centroid-0', 'centroid-1'], axis = 1)
                fts = pp[pp['cellLabelInImage'] == cell_label].iloc[:,9:].values.flatten().tolist()
                
            if self.morphology_model == False:
                fts = self.patient_protein[self.patient_protein['cellLabelInImage'] == cell_label].iloc[:,9:36+9].values.flatten().tolist()
            if len(fts) == 0:
                continue
            i_neigh = neighbours[indptr_neigh[i]:indptr_neigh[i+1]]
            self.node_ft.append(fts)
            #print('i: %d, i_neigh:'  %i, i_neigh)
            for cell in i_neigh:
                pair = np.array([i, cell])
                edge.append(pair)
        edges = np.asarray(edge).T


        edge_index = torch.tensor(edges, dtype=torch.long)
        if self.morphology_model:
            x = torch.tensor(np.array(self.node_ft).reshape(-1,36+12), dtype=torch.float)
        else:
            x = torch.tensor(np.array(self.node_ft).reshape(-1,36), dtype=torch.float)

        data = Data(x=x, edge_index=edge_index.contiguous())
        if index == 22:
            index = 15
        if index == 38:
            index = 30
        y = self.meta.loc[index,self.target]
        return data, torch.tensor(y)
    

# ds = graphDS(test = True, m = False)
# train_loader = DataLoader(ds, batch_size = 1)
# d_train= {}
# for i,p in enumerate(train_loader):
#     d_train[i] = {'x':p[0], 'y':p[1]}

    
#ds = graphDS(test = False,test_patient = 0, m = True)
#test_loader = DataLoader(ds, batch_size = 1)
#d_test = {}
#for i,p in enumerate(test_loader):
#    d_test[i] = {'x':p[0], 'y':p[1]}