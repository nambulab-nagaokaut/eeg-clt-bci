"""
LOSO_Main_Aug.py

Main training and evaluation loop for LOSO experiments on BCI2a, BCI2b, and Physionet datasets, with data augmentation.

"""

import os

gpus = [1]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
import datetime
import random
import time
import traceback

from Additional_Func import apply_max_norm, get_parameters_by_layer_type
from Load_data import LOSO
from Model.CLT.CLT import CombinedModule
from Model.CTNet.CLTNet import EEGLTransformer
from Model.CTNet.CTNet import EEGTransformer
from Model.Conformer import Conformer
from Model.EEGNet import EEGNET
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import scipy.io
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchinfo import summary
import torchvision.transforms as transforms

print(torch.__version__)
print(torch.cuda.current_device())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Configuration file
config = OmegaConf.load("./programs/Config/BCI_2a_LOSO.yaml")
# config = OmegaConf.load("./programs/Config/BCI_2b_LOSO.yaml")
OmegaConf.resolve(config)

num_augments = config.Augmentation.num_augments
n_segments = config.Augmentation.n_segments
segment_length = config.Augmentation.segment_length

batch_size = config.Training.batch_size

criterion_cls = torch.nn.CrossEntropyLoss().cuda()

dataset = config.Dataset.name
data_path = "./data/{}_gdf/".format(dataset)

# --- Choose Model: CLT, EEGNet, Conformer ----
# Model_name = "CLT"
Model_name = "EEGNet"
# Model_name = "Conformer"
# Model_name = "CTNet"
# Model_name = "CLTNet"
print("Model name: ", Model_name)


def log_model(model):
    log_model = open(save_root + "log_model.txt", "w", encoding="UTF-8")

    model_summary = summary(
        model,
        (1, config.Dataset.EEGchannels, config.Dataset.samples),
        verbose=2,
        col_width=16,
        col_names=["kernel_size", "output_size", "num_params"],
        row_settings=["var_names"],
    )
    log_model.write(str(model_summary))
    log_model.close()


def augment_torch(
    all_x, all_y, seed, n_segments, segment_length, batch_size, num_classes
):

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    C = all_x.shape[1]
    L = n_segments * segment_length
    assert all_x.shape[2] == L

    x4 = all_x.view(-1, C, n_segments, segment_length)

    per_cls = batch_size // num_classes
    aug_chunks = []
    aug_labels = []

    dev = all_x.device

    for c in range(num_classes):
        idx_c = torch.where(all_y == c)[0]
        x_c = x4[idx_c]  # (Nc, C, S, seglen)
        Nc = x_c.shape[0]

        rand = torch.randint(0, Nc, (per_cls, n_segments), device=dev)

        # flatten trick to pick (trial, segment) pairs without Python loops
        rand_flat = rand.reshape(-1)  # (per_cls*S,)
        sel = x_c[rand_flat]  # (per_cls*S, C, S, seglen)

        seg_flat = torch.arange(n_segments, device=dev).repeat(per_cls)  # (per_cls*S,)
        row = torch.arange(per_cls * n_segments, device=dev)

        # pick the matching segment for each row: -> (per_cls*S, C, seglen)
        picked = sel[row, :, seg_flat, :]

        # reshape back to (per_cls, C, L)
        aug = (
            picked.view(per_cls, n_segments, C, segment_length)
            .permute(0, 2, 1, 3)
            .reshape(per_cls, C, L)
        )

        aug_chunks.append(aug)
        aug_labels.append(torch.full((per_cls,), c, device=dev, dtype=torch.long))

    aug_data = torch.cat(aug_chunks, dim=0)  # (batch_size, C, L)
    aug_label = torch.cat(aug_labels, dim=0)  # (batch_size,)

    # shuffle
    perm = torch.randperm(aug_data.shape[0], device=dev)
    return aug_data[perm], aug_label[perm]


def augment(timg, label, seed=0, n_segments=8, segment_length=125):
    if seed is not None:
        np.random.seed(seed)

    total_length = n_segments * segment_length
    assert (
        total_length == config.Dataset.samples
    ), f"Total length {total_length} must equal config.Dataset.samples {config.Dataset.samples}"

    aug_data = []
    aug_label = []
    cls = config.Dataset.num_classes
    for cls4aug in range(cls):
        cls_idx = np.where(label == cls4aug)
        tmp_data = timg[cls_idx]
        tmp_label = label[cls_idx]

        tmp_aug_data = np.zeros(
            (int(batch_size / cls), config.Dataset.EEGchannels, config.Dataset.samples)
        )
        for ri in range(int(batch_size / cls)):
            for rj in range(n_segments):
                rand_idx = np.random.randint(0, tmp_data.shape[0], n_segments)
                tmp_aug_data[ri, :, rj * segment_length : (rj + 1) * segment_length] = (
                    tmp_data[
                        rand_idx[rj], :, rj * segment_length : (rj + 1) * segment_length
                    ]
                )

        aug_data.append(tmp_aug_data)
        aug_label.append(tmp_label[: int(batch_size / cls)])
    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    aug_shuffle = np.random.permutation(len(aug_data))
    aug_data = aug_data[aug_shuffle, :, :]
    aug_label = aug_label[aug_shuffle]

    aug_data = torch.from_numpy(aug_data).to(device).float()
    aug_label = torch.from_numpy(aug_label).to(device).long()
    return aug_data, aug_label


def get_model(model_name: str = "CLT"):
    if model_name == "CLT":
        model = CombinedModule(**config.CLT.Model_hyperparams).to(device)
    elif model_name == "EEGNet":
        model = EEGNET(**config.EEGNet.Model_hyperparams).to(device)
    elif model_name == "Conformer":
        model = Conformer(**config.EEGConformer.Model_hyperparams).to(device)
    elif model_name == "CTNet":
        model = EEGTransformer(**config.CTNet.Model_hyperparams).to(device)
    elif model_name == "CLTNet":
        model = EEGLTransformer(**config.CLTNet.Model_hyperparams).to(device)
    return model


def confusion_matrix(true_label, predict_label):
    N = torch.unique(true_label).size()[0]  # Number of classes
    conf_matrix = torch.zeros((N, N))
    for i in range(true_label.size()[0]):
        conf_matrix[true_label[i], predict_label[i]] += 1

    normalized_conf_matrix = conf_matrix / (conf_matrix.sum(dim=1, keepdim=True))
    return conf_matrix, normalized_conf_matrix


def conf_plot(conf_matrix, data_set):
    if data_set == "BCI2a":
        labels = [
            "Left hand",
            "Right hand",
            "Foot",
            "Tongue",
        ]  # i. e., 1, 2, 3, 4 corresponding to event types 769, 770, 771, 772
    elif data_set == "BCI2b":
        labels = [
            "Left hand",
            "Right hand",
        ]  # i. e., 1 and 2, corresponding to event types 769 and 770
    elif data_set == "Physionet":
        labels = ["Left fist", "Right fist", "Both fist", "Both feet"]
    plt.figure(figsize=(9, 7))
    heatmap = sns.heatmap(
        conf_matrix,
        annot=True,
        cmap="viridis",
        xticklabels=labels,
        yticklabels=labels,
        fmt=".3f",
        cbar=True,
        annot_kws={"size": 12},
    )
    if heatmap.collections[0].colorbar:
        heatmap.collections[0].colorbar.ax.tick_params(labelsize=11)
    plt.title(
        "Model {} Average Confusion Matrix of 9 Subjects".format(Model_name),
        fontsize=15,
    )
    plt.xlabel("Predicted label", fontsize=14)
    plt.ylabel("True label", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(save_root + Model_name + "_Conf_matrix.jpg")
    plt.show()


def train_val(dataset, num_augments=3, n_segments=8, segment_length=125):
    generator = torch.Generator()
    generator.manual_seed(seed_n)
    log_results = open(save_root + "Val_Acc_log.txt", "w", encoding="UTF-8")
    log_results.write("seed is " + str(seed_n) + "\n")

    try:
        Accuracies = []

        for k in range(9):  # 9 KFold iterations
            log_train = open(
                save_root + "Train_Val_log_Fold_{}.txt".format(k + 1),
                "w",
                encoding="UTF-8",
            )
            log_train.write(
                "Epoch " + "Train ACC " + "Train Loss" + "Val ACC " + "Val Loss" + "\n"
            )
            # Save hyperparams of each KFold iteration
            model_key = Model_name if Model_name != "Conformer" else "EEGConformer"
            model_hyperparams = {
                "Model_hyperparameters": config[model_key].Model_hyperparams
            }
            if Model_name == "CLT":
                model_hyperparams["Optimizer_hyperparameters"] = (
                    config.CLT.Optimizer_hyperparams
                )
            torch.save(
                model_hyperparams, save_root + "Fold_{}_best_model.pth".format(k + 1)
            )

            log_results.write("\n" + "----Fold {}----".format(k + 1) + "\n")
            train_data, train_label, val_data, val_label = LOSO(
                dataset, data_path, K=k, Train=True
            )
            All_train_data = train_data
            All_train_label = train_label

            # Turn data type to tensor
            train_data = torch.tensor(train_data, dtype=torch.float32)
            train_label = torch.tensor(train_label, dtype=torch.long)
            val_data = torch.tensor(val_data, dtype=torch.float32)
            val_label = torch.tensor(val_label, dtype=torch.long)

            # Create DataLoader
            train_loader = torch.utils.data.DataLoader(
                dataset=torch.utils.data.TensorDataset(train_data, train_label),
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
            )
            val_loader = torch.utils.data.DataLoader(
                dataset=torch.utils.data.TensorDataset(val_data, val_label),
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
            )

            # Initialize model
            model_name = Model_name
            model = get_model(model_name=model_name)

            # Optimizers
            if model_name == "CLT":
                all_params = list(model.parameters())
                EEGN_Conv_params = get_parameters_by_layer_type(
                    model.EEGN_Conv, nn.Conv2d
                )
                Linear_params = get_parameters_by_layer_type(model.Classify, nn.Linear)

                param_ids = set()
                param_ids.update(id(param) for param in EEGN_Conv_params)
                param_ids.update(id(param) for param in Linear_params)
                Other_params = [
                    param for param in all_params if id(param) not in param_ids
                ]

                optimizer = torch.optim.AdamW(
                    [
                        {
                            "params": EEGN_Conv_params,
                            "weight_decay": config.CLT.Optimizer_hyperparams[
                                "Conv_Decay"
                            ],
                        },
                        {
                            "params": Linear_params,
                            "weight_decay": config.CLT.Optimizer_hyperparams[
                                "Linear_Decay"
                            ],
                        },
                    ],
                    lr=config.Training.lr,
                    betas=(config.Training.b1, config.Training.b2),
                )

            elif model_name == "EEGNet":
                optimizer = torch.optim.AdamW(
                    params=model.parameters(),
                    lr=config.Training.lr,
                    betas=(config.Training.b1, config.Training.b2),
                )

            elif model_name == "Conformer":
                optimizer = torch.optim.AdamW(
                    params=model.parameters(),
                    lr=config.Training.lr,
                    betas=(config.Training.b1_2, config.Training.b2),
                )

            elif model_name == "CTNet":
                print("CTNet model")
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config.Training.lr,
                    betas=(config.Training.b1_2, config.Training.b2),
                )

            elif model_name == "CLTNet":
                print("CLTNet model")
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config.Training.lr,
                    betas=(config.Training.b1_2, config.Training.b2),
                )

            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.9, patience=20, min_lr=0.0001
            )

            # Train and Evaluation Loop with n_epochs
            starttime = datetime.datetime.now()
            Best_Val_acc = 0
            Best_Val_loss = 5
            counter = 0
            patience = 1000  # Epochs
            best_epoch = 0
            for e in range(config.Training.n_epochs):
                in_epoch = time.time()
                # Train Loop
                model.train()
                running_loss = 0.0
                running_corrects = 0
                total = 0
                for i, (train_data, train_label) in enumerate(train_loader):

                    All_train_data_t = torch.tensor(
                        All_train_data, dtype=torch.float32, device=device
                    )
                    All_train_label_t = torch.tensor(
                        All_train_label, dtype=torch.long, device=device
                    )

                    B = train_data.shape[0]
                    total_B = B * (1 + num_augments)

                    x = torch.empty(
                        (total_B, train_data.shape[1], train_data.shape[2]),
                        device=device,
                        dtype=train_data.dtype,
                    )
                    y = torch.empty((total_B,), device=device, dtype=train_label.dtype)

                    x[:B] = train_data
                    y[:B] = train_label

                    for aug_idx in range(num_augments):
                        current_seed = seed_n + e * 1000 + aug_idx
                        aug_x, aug_y = augment_torch(
                            All_train_data_t,
                            All_train_label_t,
                            seed=current_seed,
                            n_segments=n_segments,
                            segment_length=segment_length,
                            batch_size=B,
                            num_classes=config.Dataset.num_classes,
                        )
                        s = B * (aug_idx + 1)
                        x[s : s + B] = aug_x
                        y[s : s + B] = aug_y

                    train_data, train_label = x, y

                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(train_data)
                    loss = criterion_cls(outputs, train_label)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    # Calculate Train loss and Train acc
                    running_loss += loss.item() * train_data.size(0)
                    preds = torch.max(outputs, 1)[1]
                    running_corrects += torch.sum(preds == train_label.data)
                    total += train_label.size(0)
                epoch_loss = running_loss / total
                epoch_acc = running_corrects.double() / total

                # Evaluation Loop
                model.eval()
                val_running_loss = 0.0
                val_running_corrects = 0
                val_total = 0
                with torch.no_grad():
                    for i, (val_data, val_label) in enumerate(val_loader):

                        val_data = val_data.cuda()
                        val_label = val_label.cuda()

                        Cls = model(val_data)
                        loss_val = criterion_cls(Cls, val_label)
                        val_running_loss += loss_val.item() * val_data.size(0)
                        preds = torch.max(Cls, 1)[1]
                        val_running_corrects += torch.sum(preds == val_label.data)
                        val_total += val_label.size(0)
                val_loss = val_running_loss / val_total
                val_acc = val_running_corrects.double() / val_total
                log_train.write(
                    str(e + 1)
                    + " "
                    + str(epoch_acc.detach().cpu().numpy())
                    + " "
                    + str(epoch_loss)
                    + " "
                    + str(val_acc.detach().cpu().numpy())
                    + " "
                    + str(val_loss)
                    + "\n"
                )

                # print('Epoch:', e+1,
                #             '  Train loss: %.6f' % epoch_loss,
                #             '  Val loss: %.6f' % val_loss,
                #             '  Train accuracy %.6f' % epoch_acc,
                #             '  Val accuracy is %.6f' % val_acc)
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]["lr"]
                # print('Current learning rate:', current_lr)

                # Save best model and Early stopping
                if val_acc >= Best_Val_acc:
                    counter = 0
                    Best_Val_acc = val_acc
                    Best_Val_loss = val_loss
                    best_epoch = e + 1
                    # Save Best model
                    check_point = torch.load(
                        save_root + "Fold_{}_best_model.pth".format(k + 1)
                    )
                    check_point["model_state_dict"] = model.state_dict()
                    torch.save(
                        check_point, save_root + "Fold_{}_best_model.pth".format(k + 1)
                    )
                    print(f"--> Save Best Model at epoch {e+1}")
                else:  # Early stopping
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping at Epoch {e+1}")
                        break

            endtime = datetime.datetime.now()
            print(str(endtime - starttime))
            log_train.write(f"{e+1} Epoch runtimes: " + str(endtime - starttime))
            log_train.close()

            log_results.write(
                f"Best Model at {best_epoch}:"
                + "\n"
                + "Best Val ACC: "
                + str(Best_Val_acc.detach().cpu().numpy())
                + "       "
                + "Best Val Loss: "
                + str(Best_Val_loss)
                + "\n"
            )
            # Save Val ACC at Fold k
            print("Best Val ACC: {}".format(Best_Val_acc))
            Accuracies.append(Best_Val_acc)

        # Calculate mean of Val ACC across KFold
        mean_accuracy = np.mean([acc.detach().cpu().numpy() for acc in Accuracies])
        log_results.write("\n" + "Best Average Val ACC: " + str(mean_accuracy) + "\n")
        log_results.close()

    except Exception as e:
        print(f"Fold {k} gặp lỗi: {e}")
        raise e


def Test(dataset):
    generator = torch.Generator()
    generator.manual_seed(seed_n)
    test_result_path = save_root + "Test Results/"
    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)

    log_results = open(test_result_path + "Test_Acc_log.txt", "w", encoding="UTF-8")

    Sub_accuracies = []
    Sub_best_accuracies = []
    conf_matries = []
    for nSub in range(9):
        conf_path = test_result_path + "Confusion matrices/"
        if not os.path.exists(conf_path):
            os.makedirs(conf_path)
        log_conf = open(
            conf_path + "Subject_{}.txt".format(nSub + 1), "w", encoding="UTF-8"
        )

        log_results.write("\n" + "----Subject {}----".format(nSub + 1) + "\n")

        # check_point = torch.load(save_root + 'Fold_{}_best_model.pth'.format(nSub+1),map_location=torch.device('cuda'))
        check_point = torch.load(
            save_root + "Fold_{}_best_model.pth".format(nSub + 1),
            map_location=torch.device("cuda"),
        )

        model = get_model(model_name=Model_name)
        model.load_state_dict(check_point["model_state_dict"])
        test_data, test_label = LOSO(dataset, root_path=data_path, K=nSub, Train=False)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        test_label = torch.tensor(test_label, dtype=torch.long)

        Accuracies = []

        # Create DataLoader #144 #120
        test_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(test_data, test_label),
            batch_size=144,
            shuffle=True,
            generator=generator,
        )

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            True_label = []
            Predicted_label = []
            for data, targets in test_loader:
                data, targets = data.cuda(), targets.cuda()
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                True_label.append(targets)
                Predicted_label.append(predicted)
            True_label = torch.cat(True_label, dim=0)
            Predicted_label = torch.cat(Predicted_label, dim=0)

        accuracy = correct / total
        conf_matrix, norm_conf_matrix = confusion_matrix(
            true_label=True_label, predict_label=Predicted_label
        )

        conf_matries.append(norm_conf_matrix)
        Sub_accuracies.append(accuracy)

        print(f"Test ACC: {accuracy * 100:.2f}%")
        log_results.write("Test Acc: {}".format(accuracy) + "\n")

        log_conf.write(
            "Test ACC: "
            + str(accuracy)
            + "\n"
            + str(conf_matrix.detach().cpu().numpy())
        )
        log_conf.close()

    ave_conf = np.round(np.mean(conf_matries, axis=0), 4)
    ave_acc = np.mean([acc for acc in Sub_accuracies])
    print("\n" + f"Average Acc: {ave_acc* 100:.2f}%")
    conf_plot(ave_conf, data_set=dataset)
    log_results.write("\n" + "-------------------" + "\n")
    log_results.write("Average Test Acc across 9 subjects: {}".format(ave_acc) + "\n")
    log_results.close()


if __name__ == "__main__":

    seed_list = [1, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]

    for idx, seed in enumerate(seed_list):
        print(f"\n========== Running experiment {idx+1}/10 with seed {seed} ==========")

        seed_n = seed
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)

        save_root = f"./results/LOSO_results_replicate/{dataset}_train_val_results/Model_{Model_name}/aug_{num_augments}/seed_{seed}/"

        if not os.path.exists(save_root):
            os.makedirs(save_root)

        try:
            # train_val(dataset)
            train_val(
                dataset,
                num_augments=num_augments,
                n_segments=n_segments,
                segment_length=segment_length,
            )

            Test(dataset)
        except Exception as e:
            with open(f"./results/error_log_seed_{seed}.txt", "w") as f:
                f.write(traceback.format_exc())
            print(f"[ERROR] seed {seed}: error logged")
