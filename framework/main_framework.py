from pathlib import Path
import time

from time_counter import Clock, show_time

from nas_pipeline import NAS_pipeline
from train_pipeline import Train_pipeline
from data_pipeline import Data_Pipeline

if __name__ == "__main__":

    total_runtime_hours = 2
    total_runtime_seconds = total_runtime_hours * 60 * 60
    start_time = time.time()
    runclock = Clock(total_runtime_seconds)

    print("=== Processing Data ===")
    print("  Estimated time left:", show_time(runclock.check()))

    data_process = Data_Pipeline(
        data_info=f"{Path.home()}/Data/NAS/Nas_data/devel_dataset_0",
        augment_style="flip",
        BATCHSIZE=28,
    )

    print("=== Performing NAS ===")
    print("  Estimated time left:", show_time(runclock.check()))
    model = NAS_pipeline(
        data_process.train_loader,
        data_process.valid_loader,
        data_process.n_classes,
        data_process.n_in,
    )
    print(model)

    print("=== Training ===")
    print("  Estimated time left:", show_time(runclock.check()))
    train = Train_pipeline(model, data_process.train_loader, data_process.valid_loader)

    print("=== Predicting ===")
    print("  Estimated time left:", show_time(runclock.check()))
    test_results = train.test(data_process.test_loader)

    results = {}
    results["time_left"] = show_time(runclock.check())
    results["total_time"] = show_time(time.time() - start_time)
    results["accuracy"] = test_results[0]
    results["no_parameters"] = test_results[1]

    print(results)
