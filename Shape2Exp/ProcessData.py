import glob
import imageio
import cv2
import re
import numpy as np
import pandas as pd
import seaborn as sns
from skimage import measure
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import copy
warnings.filterwarnings("ignore")
import time
from torch.utils.data import Dataset
from sklearn import preprocessing


class CellsDataSetTNBC(Dataset):
    
    def __init__(self ,data_path,cells_data_df,
                 types_present_in_csv = False,
                 cols_to_drop = None,
                 types_data_df = None,
                 meta_data_df = None,
                 mode = 'baseline'):
        '''
        MODE : baseline - using only cell type!
               baseline_morph - using cell type and and morph
               clinic - using cell type, clinic data
               clinic_morph - using cell type, clinic data and morph
               neighbors - using cell type, clinic data neighbors
               neighbors_morph - using cell type, clinic data, neighbors and morph
        '''
        
        self.data_path = data_path
        cells = cells_data_df
        if not isinstance(cells, pd.DataFrame):
            cells = pd.read_csv(glob.glob(data_path + '/cellData.csv')[0])
        if types_data_df is not None:

            self.final = pd.merge(cells, types_data_df, how = 'inner', left_on = ['SampleID', 'cellLabelInImage'],
                                  right_on = ['Point', 'cell_idx'])
        else:
            self.final = cells

        
        self.num_pateints = len(self.final['SampleID'].unique())
        if cols_to_drop is not None:
            self.final.drop(cols_to_drop, axis = 1, inplace = True)
        points = self.final['SampleID'].unique().tolist()
        if meta_data_df is not None:
            self.meta_data = meta_data_df
        self.mode = mode
        ### props to measure
        props_list = ['label', 'centroid', 'area', 'eccentricity','major_axis_length',
                      'minor_axis_length','perimeter','equivalent_diameter_area','area_convex','extent',
                     'feret_diameter_max','orientation','perimeter_crofton','solidity', ]
        
        
        ### pre-process
        print('Welcome to prioMorph! TNBC')
        print('Building cells dicts..')
        self.cellLabelsPerPoint = {}
        self.cellIndexPerPoint = {}
        self.all_props = {}
        for i in tqdm(points):
            cl,ss = self.getCellLabels(f'Point{i}')
            self.cellLabelsPerPoint[i] = cl
            self.cellIndexPerPoint[i] = ss  
            if (self.mode.split('_')[-1] == 'morph') or (self.mode == 'neighbors') or (types_present_in_csv == False):
              self.all_props[i] = self.props(i,props_list,plot = False)
        if types_present_in_csv == False:
            all_cells = []
            patientID = self.final['SampleID'].unique().tolist()
            for p in patientID:
                try:
                    cell_types = self.all_props[p].iloc[:,[0,-1]]
                    cell_types['SampleID'] = p
                    all_cells.append(cell_types)
                except:
                    continue
            all_cells = pd.concat(all_cells, axis = 0)
            self.final = self.final.merge(all_cells, left_on = ['cellLabelInImage', 'SampleID'], right_on = ['label', 'SampleID'])
            self.final = self.final[self.final['CellType'] != 'Background']
            le = preprocessing.LabelEncoder()
            self.final['cell_label'] = le.fit_transform(self.final['CellType'])
            self.num_cells = len(self.final['cell_label'].unique())
            types = le.inverse_transform(list(range(self.num_cells)))
        else:
            types = self.final.iloc[:,-1].unique().tolist()
            self.num_cells = len(types)
        
        print(f'total annotated cell types : {self.num_cells}')
        print(f'total patients : {len(self.final.SampleID.unique())}')
        print(types)
        
    def getTypes(self, patientID):
        cell_types = []
        cell_index = self.final['cellLabelInImage'].tolist()
        for cellIndex in cell_index:
            cellLabelImage, cellIndexImage = self.cellLabelsPerPoint[patientID], self.cellIndexPerPoint[patientID]
            coords_row, coords_cols = np.where(cellIndexImage == cellIndex)
            cell_type = cellLabelImage[int(coords_row[0]), int(coords_cols[0])]
            cell_types.append(cell_type)
        return cell_types
    
    def getCellLabels(self,point):
        all_channels = glob.glob(rf'Point*/*.tiff')
        if len(all_channels) == 0:
                return None, None
        p = point.split('t')[-1]
        for img in all_channels:
            name = img.split('/')[-1].split('.')[0]
            if name == 'CellTypes':
                celltypes_lab = imageio.imread(img)
            if name in f'p{p}_labeledcellData':
                singlecell_lab = imageio.imread(img)
        return celltypes_lab,singlecell_lab


    def getCellType(self,singlecell_props,cellTypeImage):
        cell_types = []
        mapper = {0:'Background', 1:'Unknown', 2:'Endothelial', 3:'Mesenchymal', 4:'Tumor', 5:'IM-Treg', 6:'IM-CD4T', 7:'IM-CD8T',
                  8: 'IM-CD3T', 9:'IM-NK', 10:'IM-B', 11:'IM-Neutrophils', 12:'IM-Mactophages', 13:'IM-DC', 14:'IM-DC/Mono', 
                  15:'IM-Mono/Neu', 16:'IM-Other'}
        rows_list,col_list = singlecell_props['centroid-0'].tolist(), singlecell_props['centroid-1'].tolist()
        
        for row,col in zip(rows_list,col_list):
            cell_type = cellTypeImage[int(row), int(col)]
            cell_types.append(mapper[cell_type])
        singlecell_props['CellType'] = cell_types    
        return singlecell_props
            
    def props(self,point = 1, props = ('label', 'centroid', 'area', 'eccentricity'), plot = True):
        # get celltype_label, singlecell_label
        cl = self.cellLabelsPerPoint[point]
        ss = self.cellIndexPerPoint[point]
        if cl is None:
            return None
        props_single_cell = pd.DataFrame(measure.regionprops_table(ss, properties = props))
        final_props = self.getCellType(props_single_cell, cl)
        # props for each cell by type
        # final_props = final_props.loc[~final_props['CellType'].isin([0])]
        if plot:
            fig,ax = plt.subplots()
            final_props.boxplot(column=['area'], by='CellType', ax = ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_ylim([0, 2000])
            ############################################################
            fig2,ax2 = plt.subplots()
            final_props.boxplot(column=['eccentricity'], by='CellType', ax = ax2)
            ax2.set_xticklabels(ax.get_xticklabels(), rotation=90)
        return final_props

    def assignPatientVector(self,pateintID):
        try:
            onehot = self.meta_data.loc[pateintID].values
        except:
            ### no meta-data on patient.. filling 0's..
            pateintID -= 1
            onehot = np.zeros(shape = self.num_pateints+1)   
            onehot[pateintID] = 1
        return onehot
            
    def CellTypeVector(self,cellType):
        cellType -= 1
        # 0's vector in size of cell types
        onehot = np.zeros(shape = self.num_cells+1)   
        onehot[cellType] = 1
        return onehot
    def OneHotNeighbors(self, neighbors_cell_type):
        n = np.zeros(17)
        n[neighbors_cell_type] = 1
        return n
    
    
    def view_neighbors(self, patientID, cellIndex):
        all_props = self.all_props[patientID]
        props = all_props[all_props['label'] == cellIndex].values.ravel()[:-1]
        row_cord, col_cord = int(props[1]), int(props[2])
        props = props[3:]
        x0 = row_cord - 100 if row_cord - 100 > 0 else 0
        y0 = col_cord - 100 if col_cord - 100 > 0 else 0
        x1 = row_cord + 100 if row_cord + 100 < 2048 else 2048
        y1 = col_cord + 100 if col_cord + 100 < 2048 else 2048
        x = 100 if col_cord - 100 > 0 else col_cord
        y = 100 if row_cord - 100 > 0 else row_cord
        array = self.cellLabelsPerPoint[patientID][x0:x1,y0:y1]#[x0:row_cord + 300,y0:col_cord + 300]
        radius = int(2.5 * 28) ## 2.5 micron per 1 pixel. we want radius of 70 microns, so 2.5 * 28 pixels
        mask = np.zeros(array.shape)
        mask = cv2.circle(mask, (x,y), radius,color = (1, 0, 0), thickness = -1)
        out = mask * array
        #out = out[out != 0]
        plt.scatter([x], [y], c = 'k')
        plt.imshow(out, cmap = 'rainbow')

        
        
    def __len__(self):
        return self.final.shape[0]
    
    
    def __getitem__(self, index):
        #cell_time = time.time()
        cell_of_intrest = self.final.iloc[index, :]
        y = cell_of_intrest.iloc[2:-3].astype(float)
        patientID = int(cell_of_intrest['SampleID'])
        cellIndex = int(cell_of_intrest['cellLabelInImage'])
        cell_type = int(cell_of_intrest.iloc[-1])
        cellType_vector = self.CellTypeVector(cell_type)
        #finish = time.time()
        #print(f'cell type took {finish - cell_time}')
        if self.mode == 'baseline':
          return np.concatenate([cellType_vector], axis = 0).astype(float), y.values, (patientID,cellIndex)
        if self.mode == 'baseline_morph':
          batch_vector = self.assignPatientVector(patientID)
          all_props = self.all_props[patientID]
          props = all_props[all_props['label'] == cellIndex].values.ravel()[:-1]
          #row_cord, col_cord = int(props[1]), int(props[2])
          props = props[3:]
          return np.concatenate([cellType_vector, props], axis = 0).astype(float), y.values, (patientID,cellIndex)
        if self.mode == 'clinic':
          #batch_v = time.time()
          batch_vector = self.assignPatientVector(patientID)
          return np.concatenate([cellType_vector,batch_vector], axis = 0).astype(float), y.values, (patientID,cellIndex)
        if self.mode == 'clinic_morph':
          batch_vector = self.assignPatientVector(patientID)
          all_props = self.all_props[patientID]
          props = all_props[all_props['label'] == cellIndex].values.ravel()[:-1]
          #row_cord, col_cord = int(props[1]), int(props[2])
          props = props[3:]
          return np.concatenate([cellType_vector,batch_vector,props], axis = 0).astype(float), y.values, (patientID,cellIndex)
        if self.mode == 'neighbors':
          batch_vector = self.assignPatientVector(patientID)
          all_props = self.all_props[patientID]
          props = all_props[all_props['label'] == cellIndex].values.ravel()[:-1]
          row_cord, col_cord = int(props[1]), int(props[2])
          props = props[3:]
          x0 = row_cord - 100 if row_cord - 100 > 0 else 0
          y0 = col_cord - 100 if col_cord - 100 > 0 else 0
          x1 = row_cord + 100 if row_cord + 100 < 2048 else 2048
          y1 = col_cord + 100 if col_cord + 100 < 2048 else 2048
          x_c = 100 if col_cord - 100 > 0 else col_cord
          y_c = 100 if row_cord - 100 > 0 else row_cord
          array = self.cellLabelsPerPoint[patientID][x0:x1,y0:y1]#[x0:row_cord + 300,y0:col_cord + 300]
          radius = int(2.5 * 28) ## 2.5 micron per 1 pixel. we want radius of 70 microns, so 2.5 * 28 pixels
          mask = np.zeros(array.shape)
          mask = cv2.circle(mask, (x_c,y_c), radius,color = (1, 0, 0), thickness = -1)
          out = mask * array
          out = out[out != 0]
          cell_type_neighbors = np.unique(out).astype(int)
        ### get number of cells in the neighborhood
          num_cells = self.cellIndexPerPoint[patientID][x0:x1,y0:y1]
          num_out = mask * num_cells
          num_cells = [len(np.unique(num_out).astype(int))]
          onehot_neighbors = self.OneHotNeighbors(cell_type_neighbors)
          return np.concatenate([cellType_vector,batch_vector,onehot_neighbors, num_cells], axis = 0).astype(float), y.values, (patientID,cellIndex)
        if self.mode == 'neighbors_morph':
          batch_vector = self.assignPatientVector(patientID)
          all_props = self.all_props[patientID]
          props = all_props[all_props['label'] == cellIndex].values.ravel()[:-1]
          row_cord, col_cord = int(props[1]), int(props[2])
          props = props[3:]
          x0 = row_cord - 100 if row_cord - 100 > 0 else 0
          y0 = col_cord - 100 if col_cord - 100 > 0 else 0
          x1 = row_cord + 100 if row_cord + 100 < 2048 else 2048
          y1 = col_cord + 100 if col_cord + 100 < 2048 else 2048
          x_c = 100 if col_cord - 100 > 0 else col_cord
          y_c = 100 if row_cord - 100 > 0 else row_cord
          array = self.cellLabelsPerPoint[patientID][x0:x1,y0:y1]#[x0:row_cord + 300,y0:col_cord + 300]
          radius = int(2.5 * 28) ## 2.5 micron per 1 pixel. we want radius of 70 microns, so 2.5 * 28 pixels
          mask = np.zeros(array.shape)
          mask = cv2.circle(mask, (x_c,y_c), radius,color = (1, 0, 0), thickness = -1)
          out = mask * array
          out = out[out != 0]
          cell_type_neighbors = np.unique(out).astype(int)
        ### get number of cells in the neighborhood
          num_cells = self.cellIndexPerPoint[patientID][x0:x1,y0:y1]
          num_out = mask * num_cells
          num_cells = [len(np.unique(num_out).astype(int))]
          onehot_neighbors = self.OneHotNeighbors(cell_type_neighbors)
          return np.concatenate([cellType_vector,batch_vector,onehot_neighbors, num_cells, props], axis = 0).astype(float), y.values, (patientID,cellIndex)
        
        

class CellsDataSetTB(Dataset):
    def __init__(self, data_path, sc_csv, mode):
        '''
        sc_csv : path to the allTB-sarcoid data
        mode : which type od dataset to build.
                baseline : only cell type as input
                baseline_morph : baseline + morph fts
                baselineID : cell type + patient ID as input
                baselineID_morph : baselineID + morph features
                neighbors : celltype+patientID+neighbors
                neighbors_morph : neighbors + morph
        '''
        
       #reduce the sc data only to patients with segmentation mask for morph features extraction : 
        self.mode = mode
        self.data_path = data_path
        le = preprocessing.LabelEncoder()
        df = sc_csv
        
        print('Welcome to prioMorph! TB')
        print('Building cells dicts..')

        self.segmentations_paths = glob.glob(f'{self.data_path}/spatialLDA_input/segmentation_masks/*.tif')
        self.names = []
        for i in self.segmentations_paths:
            match = re.findall(r'\d+', i)
            self.names.append(int(match[0]))
        self.final = df[df['SampleID'].isin(self.names)]
        self.final['cellType'] = le.fit_transform(self.final['cell_type'])
        self.totalCellTypes = self.final['cellType'].max()
        self.totalPatients = self.final['SampleID'].nunique()
        print(f'total annotated cell types : {self.totalCellTypes}')
        print(f'total patients : {self.totalPatients}')
        print(le.inverse_transform(list(range(20))))
        if (self.mode.split('_')[-1] == 'morph') or (self.mode == 'neighbors'):
            self.final_props = self.get_props()
        
        
        
    def get_props(self):
        props_list = ['label', 'centroid', 'area', 'eccentricity','major_axis_length',
                      'minor_axis_length','perimeter','equivalent_diameter_area','euler_number','extent',
                     'feret_diameter_max','orientation','perimeter_crofton','solidity']
        final_props = {}
        self.cellIndexPerPatient = {}
        for name, img in tqdm(zip(self.names, self.segmentations_paths), total = len(self.names)):
            im = imageio.imread(img)
            self.cellIndexPerPatient[name] = im
            props_single_cell = pd.DataFrame(measure.regionprops_table(im, properties = props_list))
            final_props[int(name)] = props_single_cell            
        return final_props
              
    def CellTypeVector(self,cellType):
        # 0's vector in size of cell types
        cellType -= 1
        onehot = np.zeros(shape = self.totalCellTypes)   
        onehot[cellType] = 1
        return onehot
              
    def oneHotPatient(self,SampleID):
        # 0's vector in size of cell types
        patient_index = np.argmax(self.final['SampleID'].unique() == SampleID)
        #patient_index = int(self.final[self.final['SampleID'] == SampleID].index)
        onehot = np.zeros(shape = self.totalPatients)   
        onehot[patient_index] = 1
        return onehot
    def OneHotNeighbors(self, cell_type_neighbors):
        cells = np.zeros(self.totalCellTypes + 1)
        cells[cell_type_neighbors] = 1
        return cells
   
    def view_neighbors(self, patientID, cellIndex):
        all_props = self.final_props[patientID]
        props = all_props[all_props['label'] == cellIndex].values.ravel()[:-1]
        row_cord, col_cord = int(props[1]), int(props[2])
        props = props[3:]
        x0 = row_cord - 100 if row_cord - 100 > 0 else 0
        y0 = col_cord - 100 if col_cord - 100 > 0 else 0
        x1 = row_cord + 100 if row_cord + 100 < 2048 else 2048
        y1 = col_cord + 100 if col_cord + 100 < 2048 else 2048
        x = 100 if col_cord - 100 > 0 else col_cord
        y = 100 if row_cord - 100 > 0 else row_cord
        array = self.cellIndexPerPatient[patientID][x0:x1,y0:y1]#[x0:row_cord + 300,y0:col_cord + 300]
        radius = int(2.5 * 28) ## 2.5 micron per 1 pixel. we want radius of 70 microns, so 2.5 * 28 pixels
        mask = np.zeros(array.shape)
        mask = cv2.circle(mask, (x,y), radius,color = (1, 0, 0), thickness = -1)
        out = mask * array
        #out = out[out != 0]
        plt.scatter([x], [y], c = 'k')
        plt.imshow(out, cmap = 'rainbow')
              
    def __len__(self):
        return self.final.shape[0]
    
    def __getitem__(self, index):
        cell_of_intrest = self.final.iloc[index, :]
        y = cell_of_intrest.iloc[5:-7].values.astype(float)
        cell_label = cell_of_intrest['cellLabelInImage']
        sampleID = int(cell_of_intrest['SampleID'])
        cell_index = int(cell_of_intrest['cellLabelInImage'])
        cell_type = int(cell_of_intrest.iloc[-1])
        cell_type_vector = self.CellTypeVector(cell_type)
        if self.mode == 'baseline':
              return cell_type_vector, y, (sampleID, cell_index)
        if self.mode == 'baseline_morph':
              props = self.final_props[sampleID]
              prop = props[props['label'] == cell_index].iloc[:,3:].values.flatten()
              return np.concatenate([cell_type_vector, prop], axis = 0), y, (sampleID, cell_index)
        if self.mode == 'patientID':
            one_hot_patient = self.oneHotPatient(sampleID)
            return np.concatenate([cell_type_vector, one_hot_patient], axis = 0), y, (sampleID, cell_index)
        if self.mode == 'patientID_morph':
            props = self.final_props[sampleID]
            prop = props[props['label'] == cell_index].iloc[:,3:].values.flatten()
            one_hot_patient = self.oneHotPatient(sampleID)
            return np.concatenate([cell_type_vector, one_hot_patient,prop], axis = 0), y, (sampleID, cell_index)
        if self.mode == 'neighbors':
          batch_vector = self.oneHotPatient(sampleID)
          props = self.final_props[sampleID]
          prop = props[props['label'] == cell_index].values.flatten()
          row_cord, col_cord = int(prop[1]), int(prop[2])
          props = prop[3:]
          x0 = row_cord - 100 if row_cord - 100 > 0 else 0
          y0 = col_cord - 100 if col_cord - 100 > 0 else 0
          x1 = row_cord + 100 if row_cord + 100 < 2048 else 2048
          y1 = col_cord + 100 if col_cord + 100 < 2048 else 2048
          x_c = 100 if col_cord - 100 > 0 else col_cord
          y_c = 100 if row_cord - 100 > 0 else row_cord
          array = self.cellIndexPerPatient[sampleID][x0:x1,y0:y1]#[x0:row_cord + 300,y0:col_cord + 300]
          radius = int(2.5 * 28) ## 2.5 micron per 1 pixel. we want radius of 70 microns, so 2.5 * 28 pixels
          mask = np.zeros(array.shape)
          mask = cv2.circle(mask, (x_c,y_c), radius,color = (1, 0, 0), thickness = -1)
          out = mask * array
          self.out = out[out != 0]
          ## get neghbors cell type
          cell_label_neighbors = np.unique(out).astype(int)
          cell_types_neighbors = self.final[self.final['cellType'].isin(cell_label_neighbors)]['cellType'].unique()
          ### get number of cells in the neighborhood
          p_df = self.final[self.final['SampleID'] == sampleID]
          num_cells = [len(p_df[p_df['cellLabelInImage'].isin(cell_label_neighbors)]['cellLabelInImage'].unique())]
        
          onehot_neighbors = self.OneHotNeighbors(cell_types_neighbors)
          return np.concatenate([cell_type_vector,batch_vector,onehot_neighbors, num_cells], axis = 0).astype(float), y, (sampleID, cell_index)
        if self.mode == 'neighbors_morph':
          batch_vector = self.oneHotPatient(sampleID)
          props = self.final_props[sampleID]
          prop = props[props['label'] == cell_index].values.flatten()
          row_cord, col_cord = int(prop[1]), int(prop[2])
          props = prop[3:]
          x0 = row_cord - 100 if row_cord - 100 > 0 else 0
          y0 = col_cord - 100 if col_cord - 100 > 0 else 0
          x1 = row_cord + 100 if row_cord + 100 < 2048 else 2048
          y1 = col_cord + 100 if col_cord + 100 < 2048 else 2048
          x_c = 100 if col_cord - 100 > 0 else col_cord
          y_c = 100 if row_cord - 100 > 0 else row_cord
          array = self.cellIndexPerPatient[sampleID][x0:x1,y0:y1]#[x0:row_cord + 300,y0:col_cord + 300]
          radius = int(2.5 * 28) ## 2.5 micron per 1 pixel. we want radius of 70 microns, so 2.5 * 28 pixels
          mask = np.zeros(array.shape)
          mask = cv2.circle(mask, (x_c,y_c), radius,color = (1, 0, 0), thickness = -1)
          out = mask * array
          self.out = out[out != 0]
          ## get neghbors cell type
          cell_label_neighbors = np.unique(out).astype(int)
          cell_types_neighbors = self.final[self.final['cellType'].isin(cell_label_neighbors)]['cellType'].unique()
          ### get number of cells in the neighborhood
          p_df = self.final[self.final['SampleID'] == sampleID]
          num_cells = [len(p_df[p_df['cellLabelInImage'].isin(cell_label_neighbors)]['cellLabelInImage'].unique())]
        
          onehot_neighbors = self.OneHotNeighbors(cell_types_neighbors)
          return np.concatenate([cell_type_vector,batch_vector,props,onehot_neighbors, num_cells], axis = 0).astype(float), y, (sampleID, cell_index)
            
        

class CellsDataSetTB(Dataset):
    def __init__(self, data_path, sc_csv, mode):
        '''
        sc_csv : path to the allTB-sarcoid data
        mode : which type od dataset to build.
                baseline : only cell type as input
                baseline_morph : baseline + morph fts
                baselineID : cell type + patient ID as input
                baselineID_morph : baselineID + morph features
                neighbors : celltype+patientID+neighbors
                neighbors_morph : neighbors + morph
        '''
        
       #reduce the sc data only to patients with segmentation mask for morph features extraction : 
        self.mode = mode
        self.data_path = data_path
        le = preprocessing.LabelEncoder()
        df = sc_csv
        
        print('Welcome to prioMorph! TB')
        print('Building cells dicts..')

        self.segmentations_paths = glob.glob(f'{self.data_path}/spatialLDA_input/segmentation_masks/*.tif')
        self.names = []
        for i in self.segmentations_paths:
            match = re.findall(r'\d+', i)
            self.names.append(int(match[0]))
        self.final = df[df['SampleID'].isin(self.names)]
        self.final['cellType'] = le.fit_transform(self.final['cell_type'])
        self.totalCellTypes = self.final['cellType'].max()
        self.totalPatients = self.final['SampleID'].nunique()
        print(f'total annotated cell types : {self.totalCellTypes}')
        print(f'total patients : {self.totalPatients}')
        print(le.inverse_transform(list(range(20))))
        if (self.mode.split('_')[-1] == 'morph') or (self.mode == 'neighbors'):
            self.final_props = self.get_props()
        
        
        
    def get_props(self):
        props_list = ['label', 'centroid', 'area', 'eccentricity','major_axis_length',
                      'minor_axis_length','perimeter','equivalent_diameter_area','euler_number','extent',
                     'feret_diameter_max','orientation','perimeter_crofton','solidity']
        final_props = {}
        self.cellIndexPerPatient = {}
        for name, img in tqdm(zip(self.names, self.segmentations_paths), total = len(self.names)):
            im = imageio.imread(img)
            self.cellIndexPerPatient[name] = im
            props_single_cell = pd.DataFrame(measure.regionprops_table(im, properties = props_list))
            final_props[int(name)] = props_single_cell            
        return final_props
              
    def CellTypeVector(self,cellType):
        # 0's vector in size of cell types
        cellType -= 1
        onehot = np.zeros(shape = self.totalCellTypes)   
        onehot[cellType] = 1
        return onehot
              
    def oneHotPatient(self,SampleID):
        # 0's vector in size of cell types
        patient_index = np.argmax(self.final['SampleID'].unique() == SampleID)
        #patient_index = int(self.final[self.final['SampleID'] == SampleID].index)
        onehot = np.zeros(shape = self.totalPatients)   
        onehot[patient_index] = 1
        return onehot
    def OneHotNeighbors(self, cell_type_neighbors):
        cells = np.zeros(self.totalCellTypes + 1)
        cells[cell_type_neighbors] = 1
        return cells
   
    def view_neighbors(self, patientID, cellIndex):
        all_props = self.final_props[patientID]
        props = all_props[all_props['label'] == cellIndex].values.ravel()[:-1]
        row_cord, col_cord = int(props[1]), int(props[2])
        props = props[3:]
        x0 = row_cord - 100 if row_cord - 100 > 0 else 0
        y0 = col_cord - 100 if col_cord - 100 > 0 else 0
        x1 = row_cord + 100 if row_cord + 100 < 2048 else 2048
        y1 = col_cord + 100 if col_cord + 100 < 2048 else 2048
        x = 100 if col_cord - 100 > 0 else col_cord
        y = 100 if row_cord - 100 > 0 else row_cord
        array = self.cellIndexPerPatient[patientID][x0:x1,y0:y1]#[x0:row_cord + 300,y0:col_cord + 300]
        radius = int(2.5 * 28) ## 2.5 micron per 1 pixel. we want radius of 70 microns, so 2.5 * 28 pixels
        mask = np.zeros(array.shape)
        mask = cv2.circle(mask, (x,y), radius,color = (1, 0, 0), thickness = -1)
        out = mask * array
        #out = out[out != 0]
        plt.scatter([x], [y], c = 'k')
        plt.imshow(out, cmap = 'rainbow')
              
    def __len__(self):
        return self.final.shape[0]
    
    def __getitem__(self, index):
        cell_of_intrest = self.final.iloc[index, :]
        y = cell_of_intrest.iloc[5:-7].values.astype(float)
        cell_label = cell_of_intrest['cellLabelInImage']
        sampleID = int(cell_of_intrest['SampleID'])
        cell_index = int(cell_of_intrest['cellLabelInImage'])
        cell_type = int(cell_of_intrest.iloc[-1])
        cell_type_vector = self.CellTypeVector(cell_type)
        if self.mode == 'baseline':
              return cell_type_vector, y, (sampleID, cell_index)
        if self.mode == 'baseline_morph':
              props = self.final_props[sampleID]
              prop = props[props['label'] == cell_index].iloc[:,3:].values.flatten()
              return np.concatenate([cell_type_vector, prop], axis = 0), y, (sampleID, cell_index)
        if self.mode == 'patientID':
            one_hot_patient = self.oneHotPatient(sampleID)
            return np.concatenate([cell_type_vector, one_hot_patient], axis = 0), y, (sampleID, cell_index)
        if self.mode == 'patientID_morph':
            props = self.final_props[sampleID]
            prop = props[props['label'] == cell_index].iloc[:,3:].values.flatten()
            one_hot_patient = self.oneHotPatient(sampleID)
            return np.concatenate([cell_type_vector, one_hot_patient,prop], axis = 0), y, (sampleID, cell_index)
        if self.mode == 'neighbors':
          batch_vector = self.oneHotPatient(sampleID)
          props = self.final_props[sampleID]
          prop = props[props['label'] == cell_index].values.flatten()
          row_cord, col_cord = int(prop[1]), int(prop[2])
          props = prop[3:]
          x0 = row_cord - 100 if row_cord - 100 > 0 else 0
          y0 = col_cord - 100 if col_cord - 100 > 0 else 0
          x1 = row_cord + 100 if row_cord + 100 < 2048 else 2048
          y1 = col_cord + 100 if col_cord + 100 < 2048 else 2048
          x_c = 100 if col_cord - 100 > 0 else col_cord
          y_c = 100 if row_cord - 100 > 0 else row_cord
          array = self.cellIndexPerPatient[sampleID][x0:x1,y0:y1]#[x0:row_cord + 300,y0:col_cord + 300]
          radius = int(2.5 * 28) ## 2.5 micron per 1 pixel. we want radius of 70 microns, so 2.5 * 28 pixels
          mask = np.zeros(array.shape)
          mask = cv2.circle(mask, (x_c,y_c), radius,color = (1, 0, 0), thickness = -1)
          out = mask * array
          self.out = out[out != 0]
          ## get neghbors cell type
          cell_label_neighbors = np.unique(out).astype(int)
          cell_types_neighbors = self.final[self.final['cellType'].isin(cell_label_neighbors)]['cellType'].unique()
          ### get number of cells in the neighborhood
          p_df = self.final[self.final['SampleID'] == sampleID]
          num_cells = [len(p_df[p_df['cellLabelInImage'].isin(cell_label_neighbors)]['cellLabelInImage'].unique())]
        
          onehot_neighbors = self.OneHotNeighbors(cell_types_neighbors)
          return np.concatenate([cell_type_vector,batch_vector,onehot_neighbors, num_cells], axis = 0).astype(float), y, (sampleID, cell_index)
        if self.mode == 'neighbors_morph':
          batch_vector = self.oneHotPatient(sampleID)
          props = self.final_props[sampleID]
          prop = props[props['label'] == cell_index].values.flatten()
          row_cord, col_cord = int(prop[1]), int(prop[2])
          props = prop[3:]
          x0 = row_cord - 100 if row_cord - 100 > 0 else 0
          y0 = col_cord - 100 if col_cord - 100 > 0 else 0
          x1 = row_cord + 100 if row_cord + 100 < 2048 else 2048
          y1 = col_cord + 100 if col_cord + 100 < 2048 else 2048
          x_c = 100 if col_cord - 100 > 0 else col_cord
          y_c = 100 if row_cord - 100 > 0 else row_cord
          array = self.cellIndexPerPatient[sampleID][x0:x1,y0:y1]#[x0:row_cord + 300,y0:col_cord + 300]
          radius = int(2.5 * 28) ## 2.5 micron per 1 pixel. we want radius of 70 microns, so 2.5 * 28 pixels
          mask = np.zeros(array.shape)
          mask = cv2.circle(mask, (x_c,y_c), radius,color = (1, 0, 0), thickness = -1)
          out = mask * array
          self.out = out[out != 0]
          ## get neghbors cell type
          cell_label_neighbors = np.unique(out).astype(int)
          cell_types_neighbors = self.final[self.final['cellType'].isin(cell_label_neighbors)]['cellType'].unique()
          ### get number of cells in the neighborhood
          p_df = self.final[self.final['SampleID'] == sampleID]
          num_cells = [len(p_df[p_df['cellLabelInImage'].isin(cell_label_neighbors)]['cellLabelInImage'].unique())]
        
          onehot_neighbors = self.OneHotNeighbors(cell_types_neighbors)
          return np.concatenate([cell_type_vector,batch_vector,props,onehot_neighbors, num_cells], axis = 0).astype(float), y, (sampleID, cell_index)
    
