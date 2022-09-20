import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import optuna
from get_data import get_data
from data_augment import data_augmentation


class NAS:
    def __init__(self):
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.train_transforms, self.test_transforms = data_augmentation("default")

        (
            self.train_loader,
            self.valid_loader,
            self.test_loader,
            self.n_classes,
            self.n_in,
        ) = get_data(
            "mnist",
            "",
            28,
            self.train_transforms,
            self.test_transforms,
        )
        self.EPOCHS = 1

    def define_model(self, n_layers, in_features, dropout):

        layers = []
        layers.append(nn.Conv2d(1, in_features, kernel_size=11, padding=1))
        layers.append(nn.PReLU())

        out_features = in_features

        for i in range(n_layers):

            layers.append(
                nn.Conv2d(out_features, out_features, kernel_size=3, padding=1)
            )
            layers.append(nn.PReLU())
            p = dropout[i]
            layers.append(nn.Dropout(p))
            in_features = out_features

        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features * 216 * 216, self.n_classes))
        layers.append(nn.LogSoftmax(dim=1))

        return nn.Sequential(*layers)

    def objective(self, trial):

        n_layers = trial.suggest_int("n_layers", 1, 3)
        in_features = trial.suggest_int("in", 16, 64)

        dropout = []
        for i in range(n_layers):
            p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
            dropout.append(p)

        model = self.define_model(n_layers, in_features, dropout)
        model = model.to(self.DEVICE)

        optimizer_name = trial.suggest_categorical(
            "optimizer", ["Adam", "RMSprop", "SGD"]
        )
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        for epoch in range(self.EPOCHS):
            model.train()

            for data, target in self.train_loader:

                data, target = data.to(self.DEVICE), target.to(self.DEVICE)

                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in self.valid_loader:

                    data, target = data.to(self.DEVICE), target.to(self.DEVICE)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            accuracy = correct / len(self.valid_loader)

            trial.report(accuracy, epoch)

        return accuracy

    def search(self):

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=2)
        study.get_trials()
        trial = study.best_trial
        n_layers = trial.params["n_layers"]
        in_features = trial.params["in"]
        optimizer_name = trial.params["optimizer"]
        lr = trial.params["lr"]
        dropout = []
        for i in range(n_layers):
            p = trial.params["dropout_l{}".format(i)]
            dropout.append(p)

        model = self.define_model(n_layers, in_features, dropout)
        return model, optimizer_name, lr
