from sklearn.metrics import accuracy_score
import torch
from torch import optim
import torch.nn as nn


# from bonsai.net import Net


class Train_pipeline:
    def __init__(self, model, train_loader, valid_loader):
        self.model = model.model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer_name = model.optimizer_name
        self.lr = model.lr

        self.epochs = 1
        self.optimizer = getattr(optim, self.optimizer_name)(
            self.model.parameters(), lr=self.lr
        )

        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )

        self.trained_model = self.train()

    def get_accuracy(self, actual, predicted):
        # actual: cuda longtensor variable
        # predicted: cuda longtensor variable
        assert actual.size(0) == predicted.size(0)
        return float(actual.eq(predicted).sum()) / actual.size(0)

    def train(self):

        for epoch in range(self.epochs):

            self.model = self.model.to(self.device)
            self.model.train()
            labels, predictions = [], []
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model.forward(data)

                # store labels and predictions to compute accuracy
                labels += target.cpu().tolist()
                predictions += torch.argmax(output, 1).detach().cpu().tolist()

                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            train_acc = accuracy_score(labels, predictions)
            valid_acc = self.evaluate()

            print(
                "\tEpoch {:>3}/{:<3} | Train Acc: {:>6.2f}% | Valid Acc: {:>6.2f}% |".format(
                    epoch + 1,
                    self.epochs,
                    train_acc * 100,
                    valid_acc * 100,
                )
            )
        return self.model

    def evaluate(self):

        self.model.eval()
        labels, predictions = [], []
        for data, target in self.valid_loader:
            data = data.to(self.device)
            output = self.model.forward(data)
            labels += target.cpu().tolist()
            predictions += torch.argmax(output, 1).detach().cpu().tolist()
        return accuracy_score(labels, predictions)

    def test(self, test_loader):

        self.model.to(self.device)
        self.model.eval()
        param = sum([x.numel() for x in self.model.parameters()])
        accuracy = 0
        for data, target in test_loader:
            data = data.to(self.device)
            output = self.model.forward(data)
            predictions = torch.argmax(output, 1).detach().cpu()
            accuracy += self.get_accuracy(target, predictions)

        test_accuracy = accuracy / len(test_loader)
        return test_accuracy, param
