from tqdm import tqdm
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import r2_score, mean_squared_error
import random
from sklearn.decomposition import PCA
from numpy.polynomial.polynomial import polyfit
import numpy as np
from scipy.stats import chi2
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation


def save_model1(model, optimizer = None, filename="model.pth"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        #"optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def buildDataSet(loader, testPatietntID = None):
    if testPatietntID is None:
        train_data = {}
        patients_ids = []
        for idx,(x,y,_) in enumerate(tqdm(loader)):
            train_data[idx] = {'x': x.float(), 'y': y.float()}
            patients_ids.append(_[0].detach().numpy())
        batches = list(train_data.keys()) 
        return train_data, batches, patients_ids
    else:
        train_data = {}
        test_data = {}
        for idx,(x,y,_) in enumerate(tqdm(loader)):
            test_index = _[0] == testPatietntID
            train_index = _[0] != testPatietntID
            if torch.sum(test_index) >= 1:
                test_data[idx] = {'x': x[test_index].float(), 'y': y[test_index].float()}
            if torch.sum(train_index) >= 1:
                train_data[idx] = {'x': x[train_index].float(), 'y': y[train_index].float()}
        train_batches = list(train_data.keys())
        test_batches = list(test_data.keys()) 
        return train_data, train_batches, test_data, test_batches
    
#### this func is for training when we leave one-patient out : 
def trainEval_one_epoch1(model, loader_dict, batches_list, patients_ids, test_patient_id,
                         optimizer, criterion, device, shuffle = True): 
    if shuffle:
        random.shuffle(batches_list)
    train_error = 0
    test_error = 0
    num_train_samples = 0
    num_test_samples = 0
    for batch in batches_list:
        patients = patients_ids[batch]
        train_index = patients != test_patient_id
        test_index = patients == test_patient_id
        
        train_index = torch.Tensor(train_index).bool()
        test_index = torch.Tensor(test_index).bool()
        
        
        if torch.sum(train_index) >= 1:
            x,y = loader_dict[batch]['x'][train_index].to(device), loader_dict[batch]['y'][train_index].to(device)
            model.train()
            preds = model(x)
            train_loss = criterion(y, preds)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_error +=  train_loss.item()
            num_train_samples += torch.sum(train_index).item()
        if torch.sum(test_index) >= 1:
            x,y = loader_dict[batch]['x'][test_index].to(device), loader_dict[batch]['y'][test_index].to(device)
            with torch.no_grad():
                model.eval()
                preds = model(x)
            test_loss = criterion(y, preds)
            test_error += test_loss.item()
            num_test_samples += torch.sum(test_index).item()
    return train_error / num_train_samples, test_error / num_test_samples
### this func is when we pool the cells and train_test randomly : 
def train_one_epoch(model, loader_dict, optimizer, criterion, device, mode = 'full'):
    train_error = 0
    model.train()
    for batch in loader_dict:
      x,y = loader_dict[batch]['x'].to(device), loader_dict[batch]['y'].to(device)
      if mode == 'null':
            x = x[:,:-12]
      preds = model(x)
      train_loss = criterion(y, preds)
      optimizer.zero_grad()
      train_loss.backward()
      optimizer.step()
      train_error +=  train_loss.item()
    return train_error / len(loader_dict)

def eval_one_epoch(model, loader_dict, criterion, device, mode = 'full'):
    error = 0
    model.eval()
    for batch in loader_dict:
        x,y = loader_dict[batch]['x'].to(device), loader_dict[batch]['y'].to(device)
        if mode == 'null':
            x = x[:,:-12]
        with torch.no_grad():
            preds = model(x)
        loss = criterion(y, preds)
        error += loss.item()
    return error / len(loader_dict)

def plot_loss(train_loss, eval_loss, epoches, name,e = None, save = False):
    fig, axes = plt.subplots(nrows = 1, ncols = 1)
    x_axis = range(epoches)
    axes.plot(x_axis, train_loss, label = 'train loss')
    axes.plot(x_axis, eval_loss, label = 'eval loss')
    
    axes.set(xlabel = 'Epoches', ylabel = 'Loss (MSE)')
    if e is not None:
        axes.axvline(x=e, color='k', linestyle='-')
    axes.legend()
    if save:
        fig.savefig(f'{name}.jpeg')
    
def R2_score(model,data,device,
             mode = 'full',
             patients_ids = None,
             test_patient_id = None):
    train_predicts = []
    train_trues = []
    
    test_predicts = []
    test_trues = []
    model.eval()
    if len(data) > 2:
      for batch in data:
          patients = patients_ids[batch]
          train_index = patients != test_patient_id
          test_index = patients == test_patient_id
          if np.sum(train_index) >= 1:
              x,y = data[batch]['x'][train_index].to(device), data[batch]['y'][train_index].to(device)
              if mode == 'null':
                 x = x[:,:-12]
              with torch.no_grad():
                  preds = model(x)
              train_predicts.append(preds.detach().cpu().numpy())
              train_trues.append(y.detach().cpu().numpy())
          if np.sum(test_index) >= 1:
              x,y = data[batch]['x'][test_index].to(device), data[batch]['y'][test_index].to(device)
              if mode == 'null':
                 x = x[:,:-12]
              with torch.no_grad():
                  preds = model(x)
              test_predicts.append(preds.detach().cpu().numpy())
              test_trues.append(y.detach().cpu().numpy())
    if len(data) == 2:
      train_loader, test_loader = data[0], data[1]
      for batch in train_loader:
        x,y = train_loader[batch]['x'].to(device), train_loader[batch]['y'].to(device)
        if mode == 'null':
            x = x[:,:-12]
        with torch.no_grad():
          preds = model(x)
        train_predicts.append(preds.detach().cpu().numpy())
        train_trues.append(y.detach().cpu().numpy())
      for batch in test_loader:
        x,y = test_loader[batch]['x'].to(device), test_loader[batch]['y'].to(device)
        if mode == 'null':
            x = x[:,:-12]
        with torch.no_grad():
          preds = model(x)
        test_predicts.append(preds.detach().cpu().numpy())
        test_trues.append(y.detach().cpu().numpy())
    
    r2_train = r2_score(np.concatenate(train_trues), np.concatenate(train_predicts))
    r2_test = r2_score(np.concatenate(test_trues), np.concatenate(test_predicts))
    print(f'The Train R2 score is {np.round(r2_train, 3)}')
    print(f'The Test R2 score is {np.round(r2_test, 3)}')
    return r2_train, train_trues, train_predicts, r2_test, test_trues, test_predicts
        
def plotR2(trues, predicts, r2_val, saved_fig_name, save = False):
    preds = np.concatenate(predicts)
    gt = np.concatenate(trues)
    pca = PCA(n_components=1)
    preds_pc1 = pca.fit_transform(preds)
    gt_pc1 = pca.fit_transform(gt)
    x = preds_pc1.reshape(-1)
    y = gt_pc1.reshape(-1)
    b, m = polyfit(x, y, 1)
    ## plot
    fig, axes = plt.subplots(nrows = 1, ncols = 1)
    axes.plot(x, y, '.')
    axes.plot(x, b + m * x, '-')
    axes.set(xlabel = 'Predictions', ylabel = 'Ground Truth', title = f'R2 value : {np.round(r2_val, 3)}')    
    if save:
        fig.savefig(f'{saved_fig_name}.jpeg')
 
def train_test_split(tnbc_dataset, tnbc_patients_ids):
    all_tnbc_batches = list(tnbc_dataset.keys())
    test_indices = random.sample(all_tnbc_batches[1:], int(0.2 * len(all_tnbc_batches)))

    test_data = {}
    train_data = {}

    test_patients_id = []
    train_patients_id = []

    for batch in all_tnbc_batches:
        if batch in test_indices:
            test_data[batch] = tnbc_dataset[batch]
            test_patients_id.append(tnbc_patients_ids[batch])
        else:
            train_data[batch] = tnbc_dataset[batch]
            train_patients_id.append(tnbc_patients_ids[batch])

    return train_data, train_patients_id,test_data, test_patients_id

def getPreds(models, loader_dict, device, mode = 'full'):
    if mode not in ['full', 'null']:
        return "mode must be in [full, null]"
    if not isinstance(models, dict):
        return "you must provide models in a dict! {full/null : co-responding model}"
    model = models[mode]
    predis = []
    trues = []
    model.eval()
    model.to(device)
    for batch in loader_dict:
        x,y = loader_dict[batch]['x'].to(device), loader_dict[batch]['y'].to(device)
        if mode == 'null':
            x = x[:,:-12]
        with torch.no_grad():
            preds = model(x)
        predis.append(preds.cpu().data.numpy())
        trues.append(y.cpu().data.numpy())
    return np.concatenate(trues), np.concatenate(predis)    


def getCellPreds(models,loader_dict, device,unique_cells = 16, mode = 'full'):
    if mode not in ['full', 'null']:
        return "mode musr be in [full, null]"
    if not isinstance(models, dict):
        return "you must provide models in a dict! {full/null : co-responding model}"
    model = models[mode]
    predis = {}
    trues = {}
    model.eval()
    model.to(device)
    for batch in loader_dict:
        x,y = loader_dict[batch]['x'].to(device), loader_dict[batch]['y'].to(device)
        if mode == 'null':
            x = x[:,:-12]
        for cell,proteomics in zip(x,y):
            key = np.where(cell.cpu().numpy()[:unique_cells] == 1)[0].item()
            if key not in trues:
                trues[key] = [proteomics.cpu().data.numpy()]
                predis[key] = []
            else:
                trues[key].append(proteomics.cpu().data.numpy())
            with torch.no_grad():
                preds = model(cell.reshape(1,-1))
            predis[key].append(preds.cpu().data.numpy().reshape(-1))
    return trues, predis 

def getProPreds(models, loader_dict, device, mode = 'full'):
    if mode not in ['full', 'null']:
        return "mode musr be in [full, null]"
    if not isinstance(models, dict):
        return "you must provide models in a dict! {full/null : co-responding model}"
    model = models[mode]
    predis = []
    trues = []
    model.eval()
    for batch in loader_dict:
        x,y = loader_dict[batch]['x'].to(device), loader_dict[batch]['y'].to(device)
        if mode == 'null':
            x = x[:,:-12]
        for cell,proteomics in zip(x,y):
            with torch.no_grad():
                preds = model(cell.reshape(1,-1))
            predis.append(preds.cpu().data.numpy().reshape(-1))
            trues.append(proteomics.cpu().data.numpy())
    return np.array(trues), np.array(predis)

def buildMatrix(trues, preds, improv_metric):
    if isinstance(trues, dict):
        print('Calculating cell-type specific scores')
        bbox_dict = {'cell_type':[], 'r2_val':[]}
        for cell_type in sorted(list(trues.keys())):
            pr = np.concatenate(preds[cell_type])
            tr = np.concatenate(trues[cell_type])
            if improv_metric == 'r2':
                r2 = r2_score(tr,pr,multioutput = 'variance_weighted')
                
                
            else:
                r2 = mean_squared_error(tr,pr)
            bbox_dict['cell_type'].append(cell_type)
            bbox_dict['r2_val'].append(r2)
        boxplot_df = pd.DataFrame(bbox_dict)
        return boxplot_df
    else:
        print('Calculating protein specific scores')
        bbox_dict = {'protein':[], 'r2_val':[]}
        for protein_idx in range(preds.shape[1]):
            pr = preds[:,protein_idx]
            tr = trues[:,protein_idx]
            if improv_metric == 'r2':
                r2 = r2_score(tr,pr,multioutput = 'variance_weighted')
                
            else:
                r2 = mean_squared_error(tr,pr)
            bbox_dict['protein'].append(protein_idx)
            bbox_dict['r2_val'].append(r2)
        boxplot_df = pd.DataFrame(bbox_dict)
        return boxplot_df

def getCellProteinPreds(models, loader_dict, device,improv_metric, unique_cells = 16, mode = 'full'):
    if mode not in ['full', 'null']:
        return "mode musr be in [full, null]"
    if not isinstance(models, dict):
        return "you must provide models in a dict! {full/null : co-responding model}"
    model = models[mode]
    ftrues, fpreds = getCellPreds(models, loader_dict, device, unique_cells,mode)
    total_cells = len(list(ftrues.keys()))
    total_proteins = len((list(ftrues.values())[0])[0])
    out_matrix = np.zeros((total_cells, total_proteins))
    print(out_matrix.shape)
    for cell_type in sorted(list(ftrues.keys())):
        for protein in range(total_proteins):
            true = np.array(ftrues[cell_type])[:,protein]
            pred = np.array(fpreds[cell_type])[:,protein]
            if improv_metric == 'r2':
                r2 = r2_score(true, pred,multioutput = 'variance_weighted')
                if r2 < 0:
                    r2 = 0
                out_matrix[cell_type,protein] = r2
            else:
                out_matrix[cell_type,protein] = mean_squared_error(true, pred)
    return out_matrix
    
def LR_test(null_preds, alt_preds, ground_truth):
    alt_likelihood = np.exp(-mean_squared_error(ground_truth, alt_preds))
    null_likelihood = np.exp(-mean_squared_error(ground_truth, null_preds))
    df = 1
    G = -2 * (null_likelihood - alt_likelihood)
    p_value = chi2.sf(G, df)

    print(f'Liklihood-Ratio test pval : {p_value}')
    return p_value

def screenImprovment(model, loader_dict, device, improv_metric,unique_cells = 16, improv_type = 'global'):
    improv_types = ['global', 'protein', 'cell', 'cell-protein']
    metric_types = ['r2', 'mse']
    if improv_type not in improv_types:
        return f"improv_type must be in {improv_types}"
    if improv_metric not in metric_types:
        return f"improve_metric must be in {metric_types}"
    if improv_type == 'global':
        ftrues, fpreds = getPreds(model,loader_dict, device, mode = 'full')
        ntrues, npreds = getPreds(model,loader_dict, device, mode = 'null')
        if improv_metric == 'r2':
            fr2 = r2_score(ftrues, fpreds)
            nr2 = r2_score(ntrues, npreds)
            return (fr2, nr2)
        if improv_metric == 'r2':
            fr2 = mean_squared_error(ftrues, fpreds)
            nr2 = mean_squared_error(ntrues, npreds)
            return (fr2, nr2) 
    if improv_type == 'cell':
        ftrues, fpreds = getCellPreds(model,loader_dict, device,unique_cells, mode = 'full')
        ntrues, npreds = getCellPreds(model,loader_dict, device,unique_cells, mode = 'null')
        fmatrix = buildMatrix(ftrues, fpreds, improv_metric)
        nmatrix = buildMatrix(ntrues, npreds, improv_metric)
    if improv_type == 'protein':
        ftrues, fpreds = getProPreds(model,loader_dict, device, mode = 'full')
        ntrues, npreds = getProPreds(model,loader_dict, device, mode = 'null')
        fmatrix = buildMatrix(ftrues, fpreds, improv_metric)
        nmatrix = buildMatrix(ntrues, npreds, improv_metric)
    if improv_type == 'cell-protein':
        fmatrix = getCellProteinPreds(model,loader_dict, device,improv_metric,unique_cells, mode = 'full')
        nmatrix = getCellProteinPreds(model,loader_dict, device,improv_metric,unique_cells, mode = 'null')
        delta_matrix = fmatrix - nmatrix
        return pd.DataFrame(delta_matrix)
    if improv_metric == 'r2':
        return fmatrix - nmatrix
    if improv_metric == 'mse':
        return nmatrix - fmatrix

def buildBoxPlotR2(all_trues, all_preds):
    bbox_dict = {'model':[], 'r2_val':[]}
    for mode in ['neighbors', 'neighbors_morph']:
        sp = mode.split('_')
        if len(sp) == 2:
            first = sp[0][0]
            trues, preds = all_trues[first][1], all_preds[first][1] 
        else:
            first = sp[0][0]
            trues, preds = all_trues[first][0], all_preds[first][0]
        r2 = r2_score(trues,preds,multioutput = 'variance_weighted')
            
        bbox_dict['model'].append(mode)
        bbox_dict['r2_val'].append(r2)
              
              
    boxplot_df = pd.DataFrame(bbox_dict)
    return boxplot_df

def feature_importance(model, X_test, num_target_features):
    ig = IntegratedGradients(model)
    ig_nt = NoiseTunnel(ig)
    results = {}

    for i in range(num_target_features):
        ig_nt_attr_test = ig_nt.attribute(X_test, target = i)
        ig_nt_attr_test_sum = ig_nt_attr_test.cpu().detach().numpy().sum(0)
        ig_nt_attr_test_norm_sum = ig_nt_attr_test_sum / np.linalg.norm(ig_nt_attr_test_sum, ord=1)
        results[i] = ig_nt_attr_test_norm_sum
    return pd.DataFrame(results)

def plot_importance(impotance_results_df,
                    input_fts_names,
                    target_fts_names,
                    fts_to_display = ['NucleusArea','Nucleusamplitude'],
                    save = False,
                    name = 'ft_importance_plot.png'):
    shap_d = impotance_results_df
    shap_d.columns = target_fts_names
    shap_d.index = input_fts_names
    only_status = fts_to_display
    only_st = shap_d[only_status]
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, 65))
    cmap = mcolors.ListedColormap(colors)
    ax = only_st.iloc[:-11,:].T.plot(kind = 'bar',figsize=(18,11), label = input_fts_names[:-11], colormap=cmap)
    ax.set_xticklabels(only_status, fontsize = 20)
    ax.legend(input_fts_names,loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel('Attribution')
    if save:
        plt.savefig(f'{name}', bbox_inches='tight')    