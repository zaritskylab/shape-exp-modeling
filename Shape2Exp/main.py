import sys
import pandas as pd
import utils
from ProcessData import CellsDataSetTNBC
from torch.utils.data import DataLoader
from models import SimpleLinearNet
import copy
import torch
from tqdm import tqdm

tnbc_df = pd.read_csv(r'MIBI_TNBC_CELLS.csv')
types = pd.read_csv(r'MIBI_TNBC_idx_cell_to_type.csv')
###
#tnbc_df = tnbc_df[tnbc_df['SampleID'].isin([1,2])]
###tnbc cols to drop
cols_to_drop = ['cellSize','C','Na','Si','P','Ca','Fe','Background','B7H3','OX40','CD163', 'CSF-1R',
                'Ta','Au','tumorYN','tumorCluster','Group','immuneCluster','immuneGroup']

tnbc_neighbors_data = CellsDataSetTNBC(data_path = r'.',
                                         cells_data_df = tnbc_df,
                                         types_present_in_csv = True,
                                         cols_to_drop = cols_to_drop,
                                         types_data_df = types,
                                         meta_data_df = None,
                                         mode = 'neighbors_morph')
print()
print('###'*15)
print()
## view example of a cell and it's microenv:
tnbc_neighbors_data.view_neighbors(1, 500)
## data generator
tnbc_neighbors_loader  = DataLoader(tnbc_neighbors_data, batch_size = 64, shuffle = True)
## load to disk for faster training
tnbc_dataset, tnbc_batches, tnbc_patients_ids = utils.buildDataSet(tnbc_neighbors_loader)
## train-test split
tnbc_train_data, tnbc_train_patients_id,tnbc_test_data, tnbc_test_patients_id = utils.train_test_split(tnbc_dataset, 
                                                                                                tnbc_patients_ids)

### model HP : 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 500
tnbc_lr = 4e-3

##load models :
tnbc_model_full = SimpleLinearNet(in_features = tnbc_train_data[0]['x'].shape[1], out_features = tnbc_train_data[0]['y'].shape[1]).to(device)
tnbc_model_null = SimpleLinearNet(in_features = tnbc_train_data[0]['x'].shape[1] - 12, out_features = tnbc_train_data[0]['y'].shape[1]).to(device)

### params for each model
tnbc_criterion = torch.nn.MSELoss()
tnbc_optimizer_full = torch.optim.Adam(params = tnbc_model_full.parameters(), lr = tnbc_lr)
tnbc_optimizer_null = torch.optim.Adam(params = tnbc_model_null.parameters(), lr = tnbc_lr)

### track loss
tnbc_train_loss = {'null' : [], 'full' : []}
tnbc_val_loss = {'null' : [], 'full' : []}
best_loss_full = 100
best_loss_null = 100

#### Training loop!
for epoch in tqdm(range(EPOCHS)):
    ### train tnbc model - 1 with morph fts, and one without
    train_l_full = utils.train_one_epoch(tnbc_model_full, tnbc_train_data, tnbc_optimizer_full, tnbc_criterion, device, mode = 'full')
    eval_l_full = utils.eval_one_epoch(tnbc_model_full, tnbc_train_data, tnbc_criterion, device, mode = 'full')
    if eval_l_full < best_loss_full:
        tnbc_best_wts_full = copy.deepcopy(tnbc_model_full.state_dict())
        best_loss_full = eval_l_full
        ef = epoch
    tnbc_train_loss['full'].append(train_l_full)
    tnbc_val_loss['full'].append(eval_l_full)
    ### train null model
    train_l_null = utils.train_one_epoch(tnbc_model_null, tnbc_train_data, tnbc_optimizer_null, tnbc_criterion, device, mode = 'null')
    eval_l_null = utils.eval_one_epoch(tnbc_model_null, tnbc_train_data, tnbc_criterion, device, mode = 'null')
    if eval_l_null < best_loss_null:
        tnbc_best_wts_null = copy.deepcopy(tnbc_model_null.state_dict())
        best_loss_null = eval_l_null
        en = epoch
    tnbc_train_loss['null'].append(train_l_null)
    tnbc_val_loss['null'].append(eval_l_null)
## save best models:
tnbc_model_full.load_state_dict(tnbc_best_wts_full)
tnbc_model_null.load_state_dict(tnbc_best_wts_null)
utils.save_model1(tnbc_model_full, filename = sys.argv[1])
utils.save_model1(tnbc_model_null, filename = sys.argv[2])
## create imp df
tnbc_models = {'full' : tnbc_model_full, 'null' : tnbc_model_null}
test_trues_b, test_preds_b = utils.getPreds(tnbc_models, tnbc_test_data, device, mode = 'null')
test_trues_bm, test_preds_bm = utils.getPreds(tnbc_models, tnbc_test_data,device, mode = 'full')
all_trues = {'neigh' : [test_trues_b, test_trues_bm]}
all_preds = {'neigh' : [test_preds_b, test_preds_bm]}
df = utils.buildBoxPlotR2(all_trues, all_preds)
print(df)
df.to_csv('Results.csv')
importance_df = utils.feature_importance(tnbc_model_full.to(device), tnbc_train_data[0]['x'].to(device), num_target_features = 36)
importance_df.to_csv('Shape_ft_importance.csv')