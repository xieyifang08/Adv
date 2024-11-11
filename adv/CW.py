import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from utils import get_accuracy, imshow, get_pred
from torchattacks import CW
from PIL import Image



# Define the custom dataset
class ImageNetDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.labels_df.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')  # Ensure images are in RGB format
        label = self.labels_df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
data_dir = r'D:\adv\PLP\data\test100'  # Directory where the images are stored
csv_file = r'D:\adv\PLP\data\dev.csv'  # Path to the CSV file
dataset = ImageNetDataset(data_dir, csv_file, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load images and labels
images, labels = next(iter(data_loader))  # Load the first image and label
print('[Data loaded]')

device = "cuda"
model = models.alexnet(pretrained=True).to(device).eval()
acc = get_accuracy(model, [(images.to(device), labels.to(device))])
print('[Model loaded]')
print('Acc: %2.2f %%' % (acc))

# Set up PGD attack
atk = CW(model, c=1, kappa=1, steps=5, lr=0.05)
atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
print(atk)

# # Perform the attack
# # adv_images = atk(images.to(device), labels.to(device))
# #
# # # Visualize the results
# # idx = 0
# # pre = get_pred(model, adv_images[idx:idx + 1], device)
# # imshow(adv_images[idx:idx + 1], title="True:%d, Pre:%d" % (labels[idx].item(), pre))
# Save the adversarial images
output_dir = r'D:\adv\PLP\attack_CW_alexnet'  # Specify your output directory
# Perform the attack
adv_images = atk(images.to(device), labels.to(device))

# Save the adversarial images with reverse normalization
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# # Define reverse normalization to restore original image colors
# inv_normalize = transforms.Normalize(
#     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
#     std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
# )
#
# for idx in range(adv_images.size(0)):
#     adv_image = adv_images[idx].detach().cpu()  # Detach from graph and move to CPU
#     adv_image = inv_normalize(adv_image)  # Reverse normalization
#     adv_image = torch.clamp(adv_image, 0, 1)  # Ensure values are in the [0, 1] range
#     save_image(adv_image, os.path.join(output_dir, f'adv_{idx}.png'))  # Save the image
#
# print(f'Saved adversarial images to {output_dir}')
# Generate and save adversarial samples for all images
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)
for i, (images, labels) in enumerate(data_loader):
    images, labels = images.to(device), labels.to(device)
    adv_images = atk(images, labels)

    # Process each image in the batch (batch size = 1 here)
    for idx in range(adv_images.size(0)):
        adv_image = adv_images[idx].detach().cpu()  # Detach from graph and move to CPU
        adv_image = inv_normalize(adv_image)  # Reverse normalization
        adv_image = torch.clamp(adv_image, 0, 1)  # Ensure values are in the [0, 1] range
        save_image(adv_image, os.path.join(output_dir, f'{i}.png'))  # Save the image

    print(f'Saved adversarial image {i} to {output_dir}')
