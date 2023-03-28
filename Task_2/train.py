import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from metrics import PSNR, SSIM
from model import ResidualDenseNetwork as Model
from dataset import load

parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset', type=str, default='train')
parser.add_argument('--val_dataset', type=str, default='val')
parser.add_argument('--pretrained', default=False)
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=3)

parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--num_channels', type=int, default=1)
parser.add_argument('--growth_rate', type=int, default=10)
parser.add_argument('--num_features', type=int, default=12)
parser.add_argument('--num_blocks', type=int, default=7)
parser.add_argument('--num_layers', type=int, default=6)

args = vars(parser.parse_args())


def get_criterion():
    return nn.MSELoss()


def train(
        loader,
        model,
        criterion,
        optimizer,
        epochs,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        scheduler=None,
        validator=None,
):

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        print('Training:')
        model.train()
        train_psnr = []
        train_ssim = []
        for i, (clean_image, noisy_image) in enumerate(tqdm(loader)):
            noisy_image = noisy_image.to(device, dtype=torch.float)
            clean_image = clean_image.to(device, dtype=torch.float)
            optimizer.zero_grad()
            prediction = model(noisy_image)
            loss = criterion(prediction, clean_image)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            psnr = PSNR(data_range=1, reduction='mean')
            ssim = SSIM(1, data_range=1, reduction='mean')
            current_psnr = psnr(prediction, clean_image).item()
            current_ssim = ssim(prediction, clean_image).item()
            train_psnr.append(current_psnr)
            train_ssim.append(current_ssim)

        train_psnr = np.mean(train_psnr)
        train_ssim = np.mean(train_ssim)
        print("\nMean Train PSNR: {:.2f}\nMean Train SSIM: {:.2f}".format(train_psnr, train_ssim))
        if validator:
            print('Validation:')
            best_ssim = 0.0
            val_loader = validator
            val_psnr, val_ssim = validate(val_loader, model, device)
            if scheduler:
                scheduler.step(val_psnr)
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                filename_pth = 'runs/audio_denoising_ssim_{:.4f}_psnr_{:.4f}_epoch_{}.pth'.format(
                    best_ssim, val_psnr, epoch + 1)
                torch.save(model.state_dict(), filename_pth)
    return model


def validate(
        loader,
        model,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    model.eval()
    psnr = PSNR(data_range=1, reduction='mean')
    ssim = SSIM(1, data_range=1, reduction='mean')
    with torch.no_grad():
        val_psnr = []
        val_ssim = []
        val_mse = []
        for clean_image, noisy_image in tqdm(loader):
            noisy_image = noisy_image.to(device, dtype=torch.float)
            clean_image = clean_image.to(device, dtype=torch.float)
            prediction = model(noisy_image)
            current_psnr = psnr(clean_image, prediction).item()
            current_ssim = ssim(clean_image, prediction).item()
            mse = get_criterion()
            current_mse = mse(clean_image, prediction)

            val_psnr.append(current_psnr)
            val_ssim.append(current_ssim)
            val_mse.append(current_mse.cpu())

    val_psnr = np.mean(val_psnr)
    val_ssim = np.mean(val_ssim)
    val_mse = np.mean(val_mse)
    print('Val MSE:', val_mse)
    print("\nMean Test PSNR: {:.2f}\nMean Test SSIM: {:.2f}".format(val_psnr, val_ssim))
    print('-' * 50)

    return val_psnr, val_ssim


if __name__ == '__main__':

    train_dataset = args['train_dataset']
    val_dataset = args['val_dataset']
    path_to_pretrained = args['pretrained']
    batch_size = args['batch_size']
    lr = args['learning_rate']
    epochs = args['epochs']

    kernel_size = args['kernel_size']
    num_channels = args['num_channels']
    growth_rate = args['growth_rate']
    num_features = args['num_features']
    num_blocks = args['num_blocks']
    num_layers = args['num_layers']
    pretrained = args['pretrained']

    train_loader = load(train_dataset, batch_size, validate = False)
    val_loader = load(val_dataset, batch_size=1, validate=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(kernel_size, num_channels, growth_rate,
                  num_features, num_blocks, num_layers).to(device)
    #model = RFDN().to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of model parameters:', params)


    if pretrained:
        model.load_state_dict(
            torch.load(path_to_pretrained, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = get_criterion()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=0, verbose=True)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    train(
        train_loader,
        model,
        criterion,
        optimizer,
        epochs,
        device,
        scheduler=scheduler,
        validator=val_loader
    )
