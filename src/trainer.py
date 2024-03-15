import math
import json
import random
import time
from pathlib import Path
import gc

from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

from accelerate import Accelerator
import torch.nn.functional as F
from torchvision import transforms, datasets
import torch
from torch.utils.data import Dataset, DataLoader

from unet import UNet
from image_encoder import ImageAutoEncoder, getImageDecodings, getImageEncodings
from ddpm import DDPM
from text_encoder import getTextEncoding

import warnings
warnings.filterwarnings("ignore")


class ImageDataset(Dataset):
    def __init__(self, imgs_dir, ann_path, image_resize=None, random_crop=None,
                 center_crop=None, augment_horizontal_flip=False):
        super().__init__()

        dataset = json.load(open(ann_path, 'r'))
        self.img_captions = defaultdict(list)
        for ann in dataset["annotations"]:
            self.img_captions[ann["image_id"]].append(ann["caption"])

        self.imgs = list()
        for img in dataset["images"]:
            self.imgs.append((img["id"], imgs_dir + "/" + img["file_name"]))

        transforms_list = list()
        if image_resize:
            transforms_list.append(transforms.Resize(image_resize))
        if random_crop:
            transforms_list.append(transforms.RandomCrop(random_crop))
        if center_crop:
            transforms_list.append(transforms.CenterCrop(center_crop))
        if augment_horizontal_flip:
            transforms_list.append(transforms.RandomHorizontalFlip())

        transforms_list.append(transforms.ToTensor())
        # transforms_list.append(transforms.Normalize((0.5,), (0.5,)))
        self.transform = transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_id, img_path = self.imgs[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        captions = self.img_captions[img_id]
        cap_idx = random.randint(0, len(captions)-1)
        return (img, captions[cap_idx])


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


class Trainer(object):
    def __init__(self, diffusion_model, imgAE, images_dir, annotation_path, n_epochs=10, batch_size=16,
                 lr=1e-3, results_folder='../models/ms-coco/', split_batches=True, amp=False,
                 mixed_precision_type="fp16", grad_acc_steps=10, max_grad_norm=1.0, print_freq=100, 
                 image_resize=(256,256), random_crop=None, center_crop=None, augment_horizontal_flip=False):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else "no"
        )

        dataset = ImageDataset(images_dir, annotation_path, image_resize=image_resize, random_crop=random_crop,
                               center_crop=center_crop, augment_horizontal_flip=augment_horizontal_flip)
        data_loader = DataLoader(dataset, batch_size, shuffle=True, pin_memory=True)
        optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=lr)
        self.model, self.optimizer, self.data_loader = self.accelerator.prepare(diffusion_model,
                                                                                optimizer, data_loader)

        self.imgAE = imgAE
        self.time_steps = self.model.time_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.grad_acc_steps = grad_acc_steps
        self.max_grad_norm = max_grad_norm
        self.results_folder = results_folder
        self.dl_size = len(self.data_loader)
        self.print_freq = print_freq

        self.label_to_text = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
                              6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}

    def train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        start = time.time()
        with tqdm(self.data_loader, unit="train_batch", desc='Train') as train_data_loader:
            for step, data in enumerate(train_data_loader):
                text_encodings, text_mask = getTextEncoding(data[1])
                
                image_encodings = getImageEncodings(data[0], self.imgAE)

                with self.accelerator.autocast():
                    loss = self.model.loss(image_encodings, text_encodings=text_encodings, text_mask=text_mask)

                if self.grad_acc_steps > 1:
                    loss = loss / self.grad_acc_steps

                losses.update(loss.item(), self.batch_size)
                self.accelerator.backward(loss)
                self.accelerator.clip_grad_norm_(self.model.model.parameters(), self.max_grad_norm)

                if (step + 1) % self.grad_acc_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # ========== LOG INFO ==========
                if step % self.print_freq == 0 or step == self.dl_size-1:
                    print("\nEpoch: [{0}][{1}/{2}] Elapsed {remain:s} Loss: {loss.avg:.4f}\n".format(
                        epoch+1, step, self.dl_size, remain=timeSince(start, float(step+1)/self.dl_size),
                        loss=losses))

                    torch.save(self.model.model.state_dict(),
                            self.results_folder + f"/step-{step+1}.pth")

        return losses.avg

    def train(self):
        print("========== Training ==========")
        best_loss = 1e100
        for epoch in range(self.n_epochs):
            start_time = time.time()
            loss = self.train_epoch(epoch)
            elapsed = time.time() - start_time
            print(f"\nEpoch {epoch+1} - avg_train_loss: {loss:.4f} time: {elapsed:.0f}s\n")

            if loss < best_loss:
                best_loss = loss
                print(f"\nEpoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model\n")
                torch.save(self.model.model.state_dict(),
                            self.results_folder + f"/epoch-{epoch+1}.pth")

        torch.cuda.empty_cache()
        gc.collect()

        return best_loss

    def generateSamples(self):
        self.model.eval()
        for _, data in enumerate(self.data_loader):
            with self.accelerator.autocast():
                with torch.no_grad():
                    time_step = random.randint(0, self.time_steps-1)
                    # time_step = self.time_steps - 1
                    time_step_t = torch.full((4,), time_step, dtype=torch.long, device=self.accelerator.device)

                    image_encodings = getImageEncodings(data[0][:4], self.imgAE)
                    xt = self.model(image_encodings, time_step_t)
                    noise_decodings = getImageDecodings(xt, self.imgAE)
                    text_encodings, text_mask = getTextEncoding(data[1][:4])

                    for t in range(time_step, -1, -1):
                        tt = torch.full((4,), t, dtype=torch.long, device=self.accelerator.device)
                        _, xt = self.model.backward(xt, tt, text_encodings=text_encodings, text_mask=text_mask,
                                                    mask_prob=1.0, generate_sample=True)

                    image_decodings = getImageDecodings(xt, self.imgAE)
                    # image_decodings = xt

            _, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 3))
            for idx in range(4):
                ax = axes[0, idx]
                ax.imshow(data[0][idx].permute(1,2,0).detach().cpu().numpy())
                ax.set_title(f'Label: {data[1][idx]}')
                ax.axis('off')

            for idx in range(4):
                ax = axes[1, idx]
                ax.imshow(noise_decodings[idx].permute(1,2,0).detach().cpu().numpy())
                ax.set_title(f'Noisy input - time: {time_step}')
                ax.axis('off')

            for idx in range(4):
                ax = axes[2, idx]
                ax.imshow(image_decodings[idx].permute(1,2,0).detach().cpu().numpy())
                ax.set_title(f'Generated Images')
                ax.axis('off')

            plt.show()
            break

    def generateRandomSamples(self):
        self.model.eval()
        for _, data in enumerate(self.data_loader):
            with self.accelerator.autocast():
                with torch.no_grad():
                    time_step = self.time_steps-1
                    x_noise = torch.randn((4,8,32,32)).to(self.accelerator.device)
                    noise_decodings = getImageDecodings(x_noise, self.imgAE)
                    # noise_decodings = x_noise
                    xt = x_noise
                    text_encodings, text_mask = getTextEncoding(data[1][:4])

                    for t in range(time_step, -1, -1):
                        tt = torch.full((4,), t, dtype=torch.long, device=self.accelerator.device)
                        _, xt = self.model.backward(xt, tt, text_encodings=text_encodings, text_mask=text_mask,
                                                    mask_prob=1.0, generate_sample=True)

                    image_decodings = getImageDecodings(xt, self.imgAE)
                    # image_decodings = xt

            _, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 3))
            for idx in range(4):
                ax = axes[0, idx]
                ax.imshow(data[0][idx].permute(1,2,0).detach().cpu().numpy())
                ax.set_title(f'Label: {data[1][idx]}')
                ax.axis('off')

            for idx in range(4):
                ax = axes[1, idx]
                ax.imshow(noise_decodings[idx].permute(1,2,0).detach().cpu().numpy())
                ax.set_title(f'Noisy input - time: {time_step}')
                ax.axis('off')

            for idx in range(4):
                ax = axes[2, idx]
                ax.imshow(image_decodings[idx].permute(1,2,0).detach().cpu().numpy())
                ax.set_title(f'Generated Images')
                ax.axis('off')

            plt.show()
            break

    def encode(self):
        self.model.eval()
        for _, data in enumerate(self.data_loader):
            with self.accelerator.autocast():
                encodings = self.model.getEncoding(data[0][:4])
            break
        
        _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))

        for idx in range(4):
            ax = axes[idx]
            ax.imshow(data[0][idx].permute(1,2,0).detach().cpu().numpy().squeeze())
            ax.set_title(f'Label: {data[1][idx]}')
            ax.axis('off')

        plt.show()

        return data[0][:4], encodings

    def decode(self, encodings):
        self.model.eval()
        with self.accelerator.autocast():
            outputs = self.model.getDecoding(encodings)
        
        _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))

        for idx in range(4):
            ax = axes[idx]
            ax.imshow(outputs[idx].permute(1,2,0).detach().cpu().numpy().squeeze())
            ax.set_title(f'Generated')
            ax.axis('off')

        plt.show()

        return outputs


if __name__ == "__main__":
    
    img_dir = "../datasets/train2017"
    ann_path = "../datasets/annotations/captions_train2017.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imgAE = ImageAutoEncoder(image_channels=3, encode_dim=8, channel_mult=(1,2,4))
    imgAE_model_path = "../models/ms-coco-encoder.pth"
    imgAE.load_state_dict(torch.load(imgAE_model_path))
    imgAE = imgAE.to(device)

    unet = UNet(input_channels=8, embed_dim=128, max_seq_len=32, channel_mult=(1,2,4), 
                attn_layers=[False,False,True])
    unet_model_path = "../models/ms-coco/epoch-5.pth"
    unet.load_state_dict(torch.load(unet_model_path))

    ddpm = DDPM(unet, time_steps=300, min_beta=0.0001, max_beta=0.02)
    trainer = Trainer(ddpm, imgAE, img_dir, ann_path, n_epochs=5, batch_size=16, print_freq=500,
                      lr=1e-4, image_resize=(128,128), augment_horizontal_flip=False)
    # trainer.train()
    trainer.generateSamples()
    # trainer.generateRandomSamples()