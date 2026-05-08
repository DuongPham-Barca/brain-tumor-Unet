import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, RandomAffine, Resize, ColorJitter
import torch.nn.functional as F
import numpy as np
from torch import nn
from brain_dataset import  BrainTumorDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from unet import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from loss_function import combined_loss, dice_loss
from metric import dice_coef , iou_coef


import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = ArgumentParser(description="train faster rcnn model")
    parser.add_argument("--data_path_train", "-d", type=str, default="datasets/train", help="Path to dataset root folder")
    parser.add_argument("--data_path_val", "-v", type=str, default="datasets/val", help="Path to validation dataset folder")
    parser.add_argument("--nums_epoch", "-n", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=4)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--log_folder", "-p", type=str, default="tensorboard/brain_tumor_unet", help="path generate tensorboard log")
    parser.add_argument("--checkpoint_folder", "-c", type=str, default="trained_models",
                        help="path save trained models")
    parser.add_argument("--best_model_path", "-o", type=str, default="best_model.pt",
                        help="path to best model checkpoint")

    args = parser.parse_args()
    return args

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loss = 0
    train_dice = 0
    train_iou = 0

    train_transform = A.Compose([
        A.Resize(256, 256),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2), 
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.0, rotate_limit=15, p=0.5),
        
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, p=1),
    ], p=0.2),

        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.15),
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    train_dataset = BrainTumorDataset(
        image_dir=args.data_path_train + "/images",
        mask_dir=args.data_path_train + "/masks",
        transform=train_transform
    )

    val_dataset = BrainTumorDataset(
        image_dir=args.data_path_val + "/images",
        mask_dir=args.data_path_val + "/masks",
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True , pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    
    model = UNet(n_channels=3, n_classes=1).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if not os.path.isdir(args.log_folder):
        os.makedirs(args.log_folder)
    if not os.path.isdir(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
    writer = SummaryWriter(log_dir=args.log_folder) 

    num_iter_per_epoch = len(train_loader)
    best_val_loss = float("inf")
    start_epoch = 0

    checkpoint_path = os.path.join(args.checkpoint_folder, "last_model.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]

        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_val_loss = float("inf")
        print("No checkpoint found. Training from scratch.")

    for epoch in range(start_epoch, args.nums_epoch):
        model.train()
        progress_bar = tqdm(train_loader, colour='green')
        # progress_bar.set_description("Epoch {}/{}. Loss  {:0.4f}".format(epoch + 1, args.nums_epoch, mean_loss))
        progress_bar.set_description("Epoch {}/{}".format(epoch + 1, args.nums_epoch))
        train_loss = []
        train_dice = []
        train_iou = []
        for iter, (imgs, masks) in enumerate(progress_bar):
            imgs = imgs.to(device)
            masks = masks.to(device)
            #forward
            preds = model(imgs)
            final_losses = combined_loss(preds, masks)



            #backward
            optimizer.zero_grad()
            final_losses.backward()
            optimizer.step()

            train_loss.append(final_losses.item())
            train_dice.append(dice_coef(preds, masks).item())
            train_iou.append(iou_coef(preds, masks).item())
            progress_bar.set_description("Epoch {}/{}. Loss  {:0.4f}".format(epoch + 1, args.nums_epoch, np.mean(train_loss)))
            writer.add_scalar("loss/train", np.mean(train_loss), epoch * num_iter_per_epoch + iter)
            writer.add_scalar("dice_score/train", np.mean(train_dice), epoch * num_iter_per_epoch + iter)
            writer.add_scalar("iou_score/train", np.mean(train_iou), epoch * num_iter_per_epoch + iter)

        model.eval()
        progress_bar = tqdm(val_loader, colour='blue')
        val_loss = 0
        val_dice = 0
        val_iou = 0
        with torch.no_grad():
            for iter, (imgs, masks) in enumerate(progress_bar):
                
                imgs = imgs.to(device)
                masks = masks.to(device)  
                preds = model(imgs)
                loss = combined_loss(preds, masks)

                val_loss += loss.item()
                val_dice += dice_coef(preds, masks).item()
                val_iou += iou_coef(preds, masks).item()
                

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.best_model_path)
            print("Saved best model!")

        print(f"Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("dice_score/val", val_dice, epoch)
        writer.add_scalar("iou_score/val", val_iou, epoch)



        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_folder, "last_model.pt"))



if __name__ == '__main__':
    args = get_args()
    train(args)