import nas_search


class NAS_pipeline:
    def __init__(self, train_loader, valid_loader, n_classes, n_in):
        self.n_out = 64
        self.dropout = 0.5
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.n_classes = n_classes
        self.n_in = n_in

        self.model, self.optimizer_name, self.lr = self.nas_pipeline()

    def nas_pipeline(self):

        nas_algorithm = nas_search.NAS()
        model, optimizer_name, lr = nas_algorithm.search(
            self.train_loader,
            self.valid_loader,
            self.n_classes,
            self.n_in,
            self.n_out,
            self.dropout,
        )

        return model, optimizer_name, lr
