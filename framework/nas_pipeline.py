import nas_search

# import optuna_search


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

        # nas_algorithm = optuna_search.NAS()
        nas_algorithm = nas_search.NAS()
        model, optimizer_name, lr = nas_algorithm.search(
            self.train_loader,
            self.valid_loader,
            self.n_classes,
            self.n_in,
            self.n_out,
            self.dropout,
        )
        # model, optimizer_name, lr = nas_algorithm.search()

        return model, optimizer_name, lr


# from bonsai.nas import Bonsai


# class NAS_pipeline:
#     def __init__(self, n_classes, n_in):

#         self.hypers = {
#             "gpu_space": 0,
#             "dataset": {"name": "CIFAR10", "classes": 10},
#             "batch_size": 1,
#             "scale": 64,
#             "nodes": 7,
#             "depth": 3,
#             "patterns": [["r"]],
#             "post_patterns": 0,
#             "reduction_target": 2,
#             "lr_schedule": {"lr_max": 0.025, "T": 4},
#             "drop_prob": 0.2,
#             "prune": True,
#             "nas_schedule": {"prune_interval": 4, "cycle_len": 8},
#             "prune_rate": {"edge": 0.2, "input": 0.1},
#         }

#         self.model = self.nas_pipeline()

#     def nas_pipeline(self):

#         bonsai = Bonsai(self.hypers)
#         bonsai.generate_model()
#         print(bonsai)

#         optimizer_name = "SGD"
#         lr = 0.025

#         return bonsai.model, optimizer_name, lr
