import torch
import copy
from tqdm import tqdm

class Trainer:
    def __init__(self, model, num_features, test_patient, loader_dict, device, num_epochs=50, lr=3e-4):
        self.model = model
        self.num_features = num_features
        self.test_patient = test_patient
        self.loader_dict = loader_dict
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.best_model_wts = None
        self.epoch_loss_train = []
        self.epoch_loss_test = []
        self.best_score = float('inf')
        self.train_d = None
        self.test_d = None

    def initialize_model(self):
        self.net = self.model(self.num_features).to(self.device)

    def initialize_optimizer(self):
        self.optimizer = torch.optim.AdamW(params=self.net.parameters(), lr=self.lr)

    def initialize_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss()

    def split_data(self):
        test_patients = [self.test_patient] if not isinstance(self.test_patient, list) else self.test_patient
        train_patients = [i for i in range(1, 41) if i not in test_patients]
        self.train_d = {k: v for k, v in self.loader_dict.items() if k in train_patients}
        self.test_d = {k: v for k, v in self.loader_dict.items() if k in test_patients}

    def train_loop(self):
        self.net.train()
        lossi = 0
        for p in self.train_d:
            x, y = self.train_d[p]['x'], self.train_d[p]['y']
            x, y = x.to(self.device), y.to(self.device)
            preds = self.net(x)
            loss = self.criterion(preds, y.long().view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            lossi += loss.item()
        return lossi / len(self.train_d)

    def val_loop(self):
        self.net.eval()
        lossi = 0
        predss = []
        trues = []
        for p in self.test_d:
            x, y = self.test_d[p]['x'], self.test_d[p]['y']
            x, y = x.to(self.device), y.to(self.device)
            preds = self.net(x)
            loss = self.criterion(y.float(), preds.float().reshape(-1))
            lossi += loss.item()
            predss.append(preds.detach().cpu())
            trues.append(y.detach().cpu())
        return lossi / len(self.test_d), predss, trues

    def train(self):
        self.initialize_model()
        self.initialize_optimizer()
        self.initialize_criterion()
        self.best_model_wts = copy.deepcopy(self.net.state_dict())
        self.split_data()

        for epoch in tqdm(range(self.num_epochs), desc="Training Loop"):
            train_loss = self.train_loop()
            self.epoch_loss_train.append(train_loss)

            val_loss, predss, trues = self.val_loop()
            self.epoch_loss_test.append(val_loss)

            if val_loss < self.best_score:
                self.best_score = val_loss
                self.best_model_wts = copy.deepcopy(self.net.state_dict())

        return self.best_model_wts, self.epoch_loss_train, self.epoch_loss_test, self.test_d

class Predictor:
    def __init__(self, model, loader, device):
        self.model = model
        self.loader = loader
        self.device = device

    def predict(self):
        self.model.eval().to(self.device)
        preds = []
        trues = []
        preds_proba = []
        for p in self.loader:
            x, y = self.loader[p]['x'].to(self.device), self.loader[p]['y'].to(self.device)
            with torch.no_grad():
                pred = self.model(x)
            preds_proba.append(pred)
            preds.append(torch.argmax(pred, dim=1).item())
            trues.append(y.item())
        return preds, trues, preds_proba

def train_loop(model, num_features, test_patient, loader_dict, device, num_epochs=50, lr=3e-4):
    trainer = Trainer(model, num_features, test_patient, loader_dict, device, num_epochs, lr)
    return trainer.train()

def predict(model, loader, device):
    predictor = Predictor(model, loader, device)
    return predictor.predict()

def calculate_continuous_c_index(preds, trues):
    """
    Calculate Harrell's C-index for continuous survival time predictions.

    Parameters:
    preds (list): A list of predicted survival times.
    trues (list): A list of true survival times.

    Returns:
    float: The calculated C-index.
    """
    n = 0
    h_sum = 0

    for i in range(len(preds)):
        for j in range(len(preds)):
            if i != j:
                # Check if order of true times is consistent with predictions
                if (trues[i] > trues[j] and preds[i] > preds[j]) or (trues[i] < trues[j] and preds[i] < preds[j]):
                    h_sum += 1
                elif trues[i] == trues[j]:
                    h_sum += 0.5
                n += 1

    return h_sum / n if n != 0 else None