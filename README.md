
# Project Title

![Project Image](path_to_your_image)

## Summary

This project is based on the article [Data-modeling the interplay between single cell shape, single cell protein expression, and tissue state](https://www.biorxiv.org/content/10.1101/2024.05.29.595857v1). The study combines spatial multiplexed single-cell imaging and machine learning to explore the intricate relationships between cell shape and protein expression within human tissues. The results highlight a universal bi-directional link between cell shape and protein expression across various cell types and disease states. This research opens new avenues for understanding cellular behavior and improving disease state predictions.

## Example Usage

First, let's import the necessary modules and process the data.

```python
import pandas as pd
###
import utils
from ProcessData import CellsDataSetTNBC
tnbc_df = pd.read_csv(r'cellData.csv')
types = pd.read_csv(r'MIBI_TNBC_idx_cell_to_type.csv')
###
tnbc_df = tnbc_df[tnbc_df['SampleID'].isin([1])]
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
```
![MicroEvn_Example](https://github.com/YuvalTamir2/shape-exp-modeling/blob/main/Images/example_microenv.png)

### Model Training

Next, we'll train and eval a model using the processed data.

```python
import utils
from torch.utils.data import DataLoader
from models import SimpleLinearNet
import copy
import torch
from tqdm import tqdm
from sklearn.metrics import r2_score
## data generator
tnbc_neighbors_loader  = DataLoader(tnbc_neighbors_data, batch_size = 64, shuffle = True)
## load to disk for faster training
tnbc_dataset, tnbc_batches, tnbc_patients_ids = utils.buildDataSet(tnbc_neighbors_loader)
## train-test split
tnbc_train_data, tnbc_train_patients_id,tnbc_test_data, tnbc_test_patients_id = utils.train_test_split(tnbc_dataset, 
                                                                                                tnbc_patients_ids)

### model HP : 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 2500
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
tnbc_model_full = utils.train_eval(mode = 'full')
tnbc_model_null = utils.train_eval(model = 'null)
## save best models:
tnbc_models = {'full' : tnbc_model_full, 'null' : tnbc_model_null}
test_trues_b, test_preds_b = utils.getPreds(tnbc_models, tnbc_test_data, device, mode = 'null')
test_trues_bm, test_preds_bm = utils.getPreds(tnbc_models, tnbc_test_data,device, mode = 'full')
df = buildBoxPlotR2()
utils.plot(df)
```
![MicroEvn_Example](https://github.com/YuvalTamir2/shape-exp-modeling/blob/main/Images/example_models_compare.png)

### Model Evaluation

Finally, we'll look at the ft importance.

```python
importance_df = utils.feature_importance(tnbc_model_full.to(device), tnbc_train_data[0]['x'].to(device), num_target_features = 36)
utils.plot_importance(subset = 'Cell State')
```
![MicroEvn_Example](https://github.com/YuvalTamir2/shape-exp-modeling/blob/main/Images/example_ft_improtance.png)

For more detailed examples and explanations, please refer to the [Shape2Exp_Demo.ipynb](Shape2Exp_Demo.ipynb) notebook included in this repository.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
