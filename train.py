import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np

def main():
    """
    Main function to set up the dataset, dataloaders, and train the model.
    """
    # --- 1. Define Data Transformations ---
    # Define a transform to convert images to tensors and normalize them
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to a consistent size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # --- 2. Create Datasets ---
    data_dir = 'data/logos/datasetcopy/trainandtest'
    train_dir = f'{data_dir}/train'
    test_dir = f'{data_dir}/test'

    print("Loading datasets...")
    try:
        train_dataset = ImageFolder(root=train_dir, transform=transform)
        test_dataset = ImageFolder(root=test_dir, transform=transform)
        print("Datasets loaded successfully.")
        print(f"Found {len(train_dataset.classes)} classes in the training dataset.")
        print(f"Training dataset has {len(train_dataset)} images.")
        print(f"Test dataset has {len(test_dataset)} images.")
    except FileNotFoundError:
        print(f"Error: Make sure the data directories are correct.")
        print(f"Expected training data in: {train_dir}")
        print(f"Expected testing data in: {test_dir}")
        return
    except Exception as e:
        print(f"An error occurred while loading the datasets: {e}")
        return


    # --- 3. Create DataLoaders ---
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"DataLoaders created with batch size {batch_size}.")


    # --- 4. Visualize a Batch of Training Images (Optional) ---
    print("\nVisualizing a sample batch of training images...")
    try:
        dataiter = iter(train_loader)
        images, labels = next(dataiter)

        # Denormalize and show images
        def imshow(img):
            img = img / 2 + 0.5  # Unnormalize
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.title(f"Sample Batch")
            plt.savefig('sample_batch.png')
            print("Saved a sample batch of images to 'sample_batch.png'")

        imshow(torchvision.utils.make_grid(images[:8])) # Show first 8 images
    except Exception as e:
        print(f"Could not visualize the data. Error: {e}")
        print("This might happen if you are in an environment without a display.")
        print("A 'sample_batch.png' file will not be created.")


if __name__ == '__main__':
    main()
