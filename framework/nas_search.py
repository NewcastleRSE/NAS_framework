import torch
from torch import optim
import torch.nn as nn
import random
from model import ConvNet
from sklearn.metrics import accuracy_score


class NAS:
    def __init__(self):
        pass

    def train_nas(self, model, train_loader, valid_loader, optimizer_name, lr):

        epochs = 1
        for epoch in range(epochs):

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            model = model.to(device)
            model.train()
            optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            labels, predictions = [], []
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model.forward(data)

                # store labels and predictions to compute accuracy
                labels += target.cpu().tolist()
                predictions += torch.argmax(output, 1).detach().cpu().tolist()

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            # self.scheduler.step()

            train_acc = accuracy_score(labels, predictions)

            model.eval()
            labels, predictions = [], []
            for data, target in valid_loader:
                data = data.to(device)
                output = model.forward(data)
                labels += target.cpu().tolist()
                predictions += torch.argmax(output, 1).detach().cpu().tolist()

            valid_acc = accuracy_score(labels, predictions)

        return model, train_acc, valid_acc

    def search(self, train_loader, valid_loader, n_classes, n_in, n_out, dropout):

        nas_model = []
        acc_train = []
        acc_valid = []
        nas_optim = []
        nas_lr = []

        n_search = random.randint(1, 2)
        for i in range(0, n_search):
            rectifier_n = random.randint(0, 2)
            if rectifier_n == 0:
                rectify = nn.ReLU(inplace=True)
            elif rectifier_n == 1:
                rectify = nn.PReLU()
            else:
                rectify = nn.LeakyReLU(inplace=True)

            maxpool_n = 1
            if maxpool_n == 0:
                conv_block = random.randint(1, 2)
                maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
            else:
                conv_block = random.randint(1, 5)
                maxpool = nn.Identity()

            enc_sizes = [n_out] * conv_block

            optim_n = random.randint(0, 2)
            if optim_n == 0:
                optimizer_name = "Adam"
            elif optim_n == 1:
                optimizer_name = "RMSprop"
            else:
                optimizer_name = "SGD"

            lr = random.uniform(0.00001, 0.0001)

            print(optimizer_name)
            print(lr)

            model = ConvNet(
                n_classes, dropout, rectify, maxpool, enc_sizes, n_in, n_out
            )

            model, train_acc, valid_acc = self.train_nas(
                model, train_loader, valid_loader, optimizer_name, lr
            )

            nas_model.append(model)
            acc_train.append(train_acc)
            acc_valid.append(valid_acc)
            nas_optim.append(optimizer_name)
            nas_lr.append(lr)

        nas_dict = [
            {
                "model": nas_model,
                "train_acc": acc_train,
                "valid_acc": acc_valid,
                "optimizer": nas_optim,
                "lr": nas_lr,
            }
            for nas_model, acc_train, acc_valid, nas_optim, nas_lr in zip(
                nas_model, acc_train, acc_valid, nas_optim, nas_lr
            )
        ]

        acc_value = [m["valid_acc"] for m in nas_dict if "valid_acc" in m]
        max_acc = max(acc_value)

        for p in nas_dict:
            if p["valid_acc"] == max_acc:
                # print(p)
                final_model = p["model"]
                final_optimizer = p["optimizer"]
                final_lr = p["lr"]

        print(final_model)

        return final_model, final_optimizer, final_lr
