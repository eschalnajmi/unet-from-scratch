import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from download_and_extract import Fetcher, FetcherException
from PIL import Image, ImageSequence
from datetime import datetime
from torchmetrics.classification import Dice
import matplotlib.pyplot as plt

from unet import *
from lucchi_dataset import *

import glob



def get_data(root_dir, resource, name: str):

    compressed_file = os.path.join(root_dir, name)
    
    if not os.path.exists(compressed_file):
        Fetcher.fetch(resource, compressed_file)

def breakdownstack(files, path):
    for file in files:
        filename_ext = os.path.basename(file)
        filename = os.path.splitext(filename_ext)[0]
        try:
            im = Image.open(file)
            for i, page in enumerate(ImageSequence.Iterator(im)):
                if not os.path.isfile(f'{path}/{filename}_{i}.png'):
                    try:
                        page.save(f'{path}/{filename}_{i}.png')
                    except:
                        print(f"error - {filename}_{i}.png")
        except:
            print(f"error - {filename}")
     

def check3d(path):
    files = glob.glob(path + '/*.tif')

    if len(files) > 1:
        print(f"breaking down {files}")
        breakdownstack(files, path)
    else:
        print(f"No 3D data found in {files}")

root_dir = os.environ.get('data', "data")

if __name__ == "__main__":

    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4
    EPOCHS = 50


    for f in os.scandir(os.path.join(root_dir, "Lucchi")):
        check3d(f.path)

    timestamp = datetime.now().strftime("%d%m%Y-%H%M%S")

    DATA_PATH = os.path.join(root_dir, "Lucchi")
    MODEL_PATH = f"models/model_{timestamp}.pth"

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    train_dataset = LucchiDataset(DATA_PATH)

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(train_dataset, [0.8,0.2], generator=generator)
    

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    best_metric = -1
    best_metric_epoch = -1
    all_mean_dice = []
    all_train_loss = []

    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        idx = 0

        for img_mask in train_loader:
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            optimizer.zero_grad()

            loss = criterion(y_pred, mask)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            idx += 1

        train_loss = train_loss/idx+1
        all_train_loss.append(train_loss)

        model.eval()

        val_loss = 0

        with torch.no_grad():
            for idx, img_mask in enumerate(val_loader):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                y_pred = model(img)

                loss = criterion(y_pred, mask)
                val_loss += loss.item()

            val_loss = val_loss/idx+1

        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        metric = Dice().to(device)
        mean_dice = metric(y_pred, mask.long())
        all_mean_dice.append(mean_dice.cpu())

        if mean_dice > best_metric:
            best_metric = mean_dice
            best_metric_epoch = epoch+1
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"New best metric saved! Mean dice: {best_metric:.4f}")
        
        print("*"*50)

    print(f"Training finished! Best metric: {best_metric:.4f} at epoch: {best_metric_epoch}. Model saved as {MODEL_PATH}")
    
    plt.figure("train", (12,6))
    plt.subplot(1,2,1)
    plt.title("training loss")
    x = [i+1 for i in range(len(all_mean_dice))]
    y = all_train_loss
    plt.xlabel("epoch")
    plt.plot(x, y, color="red")

    plt.subplot(1,2,2)
    plt.title("mean dice")
    x = [1* (i+1) for i in range(len(all_mean_dice))]
    y = all_mean_dice
    plt.xlabel("epoch")
    plt.plot(x,y)

    plt.show()


    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            roi_size=(160,160)
            sw_batch_size=4
            img = val_data[0].float().to(device)
            val_outputs = model(img)

            img = img.unsqueeze(0).cpu()

            plt.figure("test", (18,6))
            plt.subplot(1, 4, 1)
            plt.title(f"image {i}")
            plt.imshow(val_data[0][0,0,:,:], cmap="gray")
            plt.subplot(1, 4, 2)
            plt.title(f"label {i}")
            plt.imshow(val_data[1][0,0,:,:])
            plt.subplot(1,4,3)
            plt.title(f"output {i}")
            plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0,:,:])
            plt.subplot(1,4,4)
            plt.title(f"masked output {i}")
            plt.imshow(val_data[0][0,0,:,:], cmap="gray")
            plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0,:,:], alpha=0.5, cmap="viridis")
            plt.show()
            if i==2:
                break

