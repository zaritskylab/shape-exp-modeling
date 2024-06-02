import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch_geometric.data import DataLoader
from sklearn.preprocessing import label_binarize

from Data import graphDS
from models import GCN
from utils import *

## HP : 
device = 'cuda' if torch.cuda.is_available else 'cpu'
print(device)
num_patients = 41
results = {}
ds = graphDS(test = False,test_patient = 0, m = True, target = 'Recurrence')
loader = DataLoader(ds)
loader_dict= {}
for idx,p in enumerate(tqdm(loader, desc="building loader")):
   idx += 1
   loader_dict[idx] = {'x':p[0], 'y':p[1]}
## leave one out CV 
for i in range(1,num_patients):
    try:
        print(f'test patient {i}')
        best_model_wts, epoch_loss_train, epoch_loss_test, test_dict = train_loop(GCN, 36+12, i, loader_dict, device)
        net = GCN(36+12)
        net.load_state_dict(best_model_wts)
        preds, trues, preds_proba = predict(net, test_dict, device)
        print(preds, trues)
        results[i] = [preds[0], trues[0], preds_proba]
    except:
        continue

y_pred_all = np.array(list(results.values()))[:,0]
y_true_all = np.array(list(results.values()))[:,1]
# Binarize the true labels
n_classes = y_pred_all.shape[1]
y_true_binarized = label_binarize(y_true_all, classes=range(n_classes))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_all[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_pred_all.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot the ROC curve for the micro-average
plt.figure(figsize=(8, 6))
plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Plot the diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro-averaged One-vs-Rest\nReceiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()