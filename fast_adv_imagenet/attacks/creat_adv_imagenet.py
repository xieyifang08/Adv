import os
import argparse
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import scipy
import tqdm
import numpy as np
from copy import deepcopy
import torch
from torch.utils import data
# from torchvision.transforms import InterpolationMode
import pandas as pd
from torchvision import transforms, models
import sys

from torchvision.transforms import InterpolationMode

from fast_adv.utils.messageUtil import send_email
from fast_adv_imagenet.utils import model_utils
from fast_adv.attacks import DDN
import warnings
from PIL import Image
import foolbox
from foolbox.criteria import Misclassification
from foolbox.distances import Linfinity
from foolbox.attacks import DeepFoolL2Attack, PGD, CarliniWagnerL2Attack

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)
parser = argparse.ArgumentParser(description='Extend sample')
parser.add_argument('--max-norm', type=float, default=10, help='max norm for the adversarial perturbations')
parser.add_argument('--img_size', default=224, type=int, help='pic size')
parser.add_argument('--data', default=r'D:\adv\PLP\data',
                    help='path to dataset')
parser.add_argument('--batch-size', '-b', default=32, type=int, help='mini-batch size')
parser.add_argument("--shape", type=int, default=None)
parser.add_argument("--DDN", type=bool, default=True)
parser.add_argument("--DeepFool", type=bool, default=True)
parser.add_argument("--PGD", type=bool, default=True)
parser.add_argument("--CW", type=bool, default=True)
# parser.add_argument("--data-loader", type=str, default='train')

args = parser.parse_args()
print(args)
path = "./adv/"

attackers = {
    'C&W': CarliniWagnerL2Attack,  # 距离无限制
    'DeepFoolAttack': DeepFoolL2Attack,  # 源码上没有限制
    'PGD': PGD,  # clip——epsilon=0.3
    'DDN': DDN,
}
# 加载到GPU
DEVICE = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')


def load_data_for_defense(csv, input_dir, img_size=args.img_size, batch_size=args.batch_size):
    jir = pd.read_csv(csv)
    all_imgs = [os.path.join(input_dir, str(i)) for i in jir['ImageId'].tolist()]
    all_labels = jir['TrueLabel'].tolist()
    dev_data = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})

    transformer = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    datasets = {
        'dev_data': ImageSet(dev_data, transformer)
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       # num_workers=8,
                       shuffle=False) for ds in datasets.keys()
    }
    return dataloaders


class ImageSet(Dataset):
    def __init__(self, df, transformer):
        self.df = df
        self.transformer = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        b = Image.open(image_path).convert('RGB')  # 使用 PIL 读取图像
        image = self.transformer(b)
        label_idx = self.df.iloc[item]['label_idx']
        sample = {
            'dataset_idx': item,
            'image': image,
            'label_idx': label_idx,
            'filename': os.path.basename(image_path)
        }
        return sample


test_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), os.path.join(args.data, 'test100'))['dev_data']

print(len(test_loader))

model = models.alexnet(pretrained=True)
# 将模型设置为评估模式
model.eval()


def attack(image, label, attack_name):
    fmodel = foolbox.models.PyTorchModel(model.eval().cuda(), bounds=(0, 1),
                                         num_classes=1000)  # , preprocessing=(mean, std)
    criterion1 = Misclassification()
    distance = Linfinity  # MeanAbsoluteDistance
    attacker = attackers[attack_name](fmodel, criterion=criterion1, distance=distance)

    image = image.cpu().numpy()
    label = label.cpu().numpy()

    adversarials = image.copy()
    advs = attacker(image, label)  # , unpack=True, steps=self.max_iter, subsample=self.subsample)
    for i in tqdm.tqdm(range(len(advs)), ncols=80):
        if advs is not None:
            adv = torch.renorm(torch.from_numpy(advs[i] - image[i]), p=2, dim=0, maxnorm=100).numpy() + image[i]

            adversarials[i] = adv
    adversarials = torch.from_numpy(adversarials).to(DEVICE)

    return adversarials


print("data_loader :", len(test_loader))
for i, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
    # if args.batch_size * i < 4990:
    #     continue
    images, labels = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE)
    print("\nepoch " + str(i) + '\n')
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    if args.DeepFool is True:
        DeepFool = attack(images, labels, 'DeepFoolAttack')
        for t in range(images.shape[0]):
            # ddn2 = np.transpose(DeepFool[t].detach().cpu().numpy(), (1, 2, 0))
            ddn2 = (DeepFool[t].detach().cpu().numpy() * 255).astype(np.uint8)  # 缩放到 [0, 255]
            ddn2 = np.transpose(ddn2, (1, 2, 0))
            name = str(args.batch_size * i + t) + '.png'
            out_path = os.path.join(path, "DeepFool")
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out = os.path.join(out_path, name)
            image = Image.fromarray(np.clip(ddn2, 0, 255).astype(np.uint8))
            image.save(out)

    if args.PGD is True:
        PGD = attack(images, labels, 'PGD')
        for t in range(images.shape[0]):
            # ddn2 = np.transpose(PGD[t].detach().cpu().numpy(), (1, 2, 0))
            ddn2 = (PGD[t].detach().cpu().numpy() * 255).astype(np.uint8)  # 缩放到 [0, 255]
            ddn2 = np.transpose(ddn2, (1, 2, 0))
            name = str(args.batch_size * i + t) + '.png'
            out_path = os.path.join(path, "PGD")
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out = os.path.join(out_path, name)
            image = Image.fromarray(np.clip(ddn2, 0, 255).astype(np.uint8))
            image.save(out)
