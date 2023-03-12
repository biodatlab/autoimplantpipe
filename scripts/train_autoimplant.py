import argparse
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    Resized,
    RandFlipd,
)
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.data import DataLoader, Dataset, decollate_batch
from autoimplantpipe import AutoImplantUNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--df_data",
        type=str,
        default="cp_defect_complete_path.csv",
        help="input csv file collected data path",
    )
    parser.add_argument(
        "--pretrain_path",
        type=str,
        default="model/AutoImplant2020_176x224x144_fold0.h5",
        help="input pre-train model path",
    )
    parser.add_argument("--batch_size", type=int, help="define batch size for training")
    parser.add_argument(
        "--n_epochs", type=int, help="define number of epoch for training"
    )
    parser.add_argument("--model_path", type=str, help="define output model path")
    args = parser.parse_args()
    return args


def create_dict_datasets(images, labels):
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(images, labels)
    ]
    return data_dicts


def create_data_loader(df_path, batch_size):
    defect_skull_path = df_path.defect_skull_path.to_list()
    complete_skull_path = df_path.complete_skull_path.to_list()
    data_dict = create_dict_datasets(defect_skull_path, complete_skull_path)

    train_files, val_files = train_test_split(data_dict, test_size=0.1, random_state=42)
    val_files, _ = train_test_split(val_files, test_size=0.5, random_state=42)

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Resized(
                keys=["image", "label"],
                spatial_size=(176, 224, 144),
                mode=["area", "nearest"],
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Resized(
                keys=["image", "label"],
                spatial_size=(176, 224, 144),
                mode=["area", "nearest"],
            ),
        ]
    )
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
    return train_ds, train_loader, val_loader


def main():
    args = parse_args()
    model_path = args.model_path
    pretrain_path = args.pretrain_path
    df_path = pd.read_csv(args.df_data)

    train_ds, train_loader, val_loader = create_data_loader(df_path, args.batch_size)
    set_determinism(seed=0)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    model = AutoImplantUNet().to(device)
    model.load_pretrained_model(pretrain_path)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    max_epochs = args.n_epoch
    wandb.init(
        entity="autoimplant",
        project="autoimplant",
        name=f"unet_autoimplant_ep{max_epochs}",
    )
    wandb.config = {"learning_rate": 0.0001, "epochs": max_epochs, "batch_size": 1}
    wandb.watch(model)

    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    post_pred = Compose([AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            train_outputs = [post_pred(i) for i in decollate_batch(outputs)]
            train_labels = [post_label(i) for i in decollate_batch(labels)]
            dice_metric(y_pred=train_outputs, y=train_labels)
            dice = dice_metric.aggregate().item()
            train_loss = loss_function(outputs, labels)
            wandb.log({"train_step_dice": dice})
            wandb.log({"train_step_loss": train_loss.item()})
            train_loss.backward()
            optimizer.step()
            epoch_loss += train_loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size},"
                f"train_loss: {train_loss.item():.4f}, "
                f"train_dice: {dice:.4f}"
            )

        train_dice = dice_metric.aggregate().item()
        dice_metric.reset()
        epoch_loss /= step
        print(
            f"epoch {epoch + 1}"
            f"\ntrain average loss: {epoch_loss:.4f}"
            f"\ntrain mean dice: {train_dice:.4f}"
        )
        wandb.log({"train_loss": epoch_loss})
        wandb.log({"train_dice": train_dice})

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    inputs, labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    outputs = model(inputs)
                    val_outputs = [post_pred(i) for i in decollate_batch(outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    val_loss = loss_function(outputs, labels)
                val_dice = dice_metric.aggregate().item()
                dice_metric.reset()
                wandb.log({"val_loss": val_loss.item()})
                wandb.log({"val_dice": val_dice})
                if val_dice > best_metric:
                    best_metric = val_dice
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), model_path)
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current val mean dice: {val_dice:.4f}"
                    f"\nbest val mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
    wandb.finish()


if __name__ == "__main__":
    main()
