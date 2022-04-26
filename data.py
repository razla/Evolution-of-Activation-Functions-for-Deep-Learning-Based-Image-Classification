from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch

def load_dataset(n_dataset, train1_size, train2_size):
    if n_dataset == 1:
        #### MNIST ####
        mean = torch.tensor([0.1306])
        std = torch.tensor([0.3081])
        training_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        )

        test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        )

    elif n_dataset == 2:
        #### Fashion-MNIST ####
        mean = torch.tensor([0.2864])
        std = torch.tensor([0.3533])
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        )

        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        )
    elif n_dataset == 3:
        #### KMNIST ####
        mean = torch.tensor([0.1918])
        std = torch.tensor([0.3484])
        training_data = datasets.KMNIST(
            root="data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        )

        test_data = datasets.KMNIST(
            root="data",
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        )
    elif n_dataset == 4:
        #### USPS ####
        mean = torch.tensor([0.2487])
        std = torch.tensor([0.2998])
        training_data = datasets.USPS(
            root="data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(28),
                transforms.Normalize(mean, std)
            ])
        )

        test_data = datasets.USPS(
            root="data",
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(28),
                transforms.Normalize(mean, std)
            ])
        )
    else:
        raise Exception("Sorry, no such dataset")

    train_size = int(len(training_data) * train1_size)
    val_size = int(len(training_data) * train2_size)
    left_overs_size = int(len(training_data) - (train_size + val_size))
    test_size = int(len(test_data))
    left_out_test_size = int(len(test_data) - test_size)

    # Splits the training set into training and validation subsets
    train_val_subset, left_overs = torch.utils.data.random_split(training_data, [train_size + val_size, left_overs_size],
                                                                 generator=torch.Generator().manual_seed(1))

    train_subset, val_subset, _ = torch.utils.data.random_split(training_data, [train_size, val_size, left_overs_size],
                                                             generator=torch.Generator().manual_seed(1))

    test_subset, _ = torch.utils.data.random_split(test_data, [test_size, left_out_test_size],
                                                   generator=torch.Generator().manual_seed(1))

    batch_size = int(len(train_val_subset) / 10)

    ######### Creating data loaders #########
    all_train_dataloader = DataLoader(dataset=train_val_subset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_subset, batch_size=batch_size, shuffle=False)

    return all_train_dataloader, train_loader, val_loader, test_loader, batch_size