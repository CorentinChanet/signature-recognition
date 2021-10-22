import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset, random_split


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def load_dataset(data_path):

    # Load all the images
    transformation = transforms.Compose([
        # Randomly augment the image data
        # Random horizontal flip
        #transforms.RandomHorizontalFlip(0.5),
        transforms.Grayscale(1),
        #transforms.RandomVerticalFlip(0.3),
        # transform to tensors
        torchvision.transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN),
        transforms.ToTensor()
    ])

    # Load all of the images, transforming them
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
    )

    train_indices, val_indices = train_test_split(list(range(len(full_dataset.targets))), test_size=0.3,
                                                  stratify=full_dataset.targets)

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    test_subset = torch.utils.data.Subset(full_dataset, val_indices)

    train_dataset = DatasetFromSubset(
        train_subset, transform=transformation)

    test_dataset = DatasetFromSubset(
        test_subset, transform=transformation)

    # define a loader for the training data we can iterate through in 50-image batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=0,
        shuffle=True
    )

    # define a loader for the testing data we can iterate through in 50-image batches
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=0,
        shuffle=True
    )

    return train_loader, test_loader


#####################################################################################################


class Net(nn.Module):

    # Defining the Constructor
    def __init__(self, num_classes=2):
        super(Net, self).__init__()

        # In the init function, we define each layer we will use in our model

        # Our images are RGB, so we have input channels = 3.
        # We will apply 12 filters in the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)

        # A second convolutional layer takes 12 input channels, and generates 24 outputs
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)

        # We in the end apply max pooling with a kernel size of 2
        self.pool = nn.MaxPool2d(kernel_size=2)

        # A drop layer deletes 20% of the features to help prevent overfitting
        self.drop = nn.Dropout2d(p=0.2)

        # Our 128x128 image tensors will be pooled twice with a kernel size of 2. 128/2/2 is 32.
        # This means that our feature tensors are now 32 x 32, and we've generated 24 of them

        # We need to flatten these in order to feed them to a fully-connected layer
        self.fc = nn.Linear(in_features=64 * 64 * 24, out_features=num_classes)

    def forward(self, x):
        # In the forward function, pass the data through the layers we defined in the init function

        # Use a ReLU activation function after layer 1 (convolution 1 and pool)
        x = F.relu(self.pool(self.conv1(x)))

        # Use a ReLU activation function after layer 2
        x = F.relu(self.pool(self.conv2(x)))

        # Select some features to drop to prevent overfitting (only drop during training)
        x = F.dropout(self.drop(x), training=self.training)

        # Flatten
        x = x.view(-1, 64 * 64 * 24)
        # Feed to fully-connected layer to predict class
        x = self.fc(x)
        # Return class probabilities via a log_softmax function
        return torch.log_softmax(x, dim=1)


def train(model, device, train_loader, epoch):
    # Define loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_criteria = nn.CrossEntropyLoss()

    # Set the model to training mode
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # Use the CPU or GPU as appropriate
        # Recall that GPU is optimized for the operations we are dealing with
        data, target = data.to(device), target.to(device)

        # Reset the optimizer
        optimizer.zero_grad()

        # Push the data forward through the model layers
        output = model(data)

        # Get the loss
        loss = loss_criteria(output, target)

        # Keep a running total
        train_loss += loss.item()

        # Backpropagate
        loss.backward()
        optimizer.step()

        # Print metrics so we see some progress
        print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))

    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx + 1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss


def test(model, device, test_loader):
    loss_criteria = nn.CrossEntropyLoss()
    # loss_criteria = nn.BCELoss()

    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)

            # Get the predicted classes for this batch
            output = model(data)

            # Calculate the loss for this batch
            test_loss += loss_criteria(output, target).item()

            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # return average loss for the epoch
    return avg_loss, correct

def main():
    # Recall that we have resized the images and saved them into
    train_folder = '/Users/Corty/Downloads/signatures_CNN/train'
    classes = ['Class_0', 'Class_1']

    # Get the iterative dataloaders for test and training data
    train_loader, test_loader = load_dataset(train_folder)
    batch_size = train_loader.batch_size
    print("Data loaders ready to read", train_folder)

    device = "cpu"

    # Create an instance of the model class and allocate it to the device
    model = Net(num_classes=len(classes)).to(device)

    print(model)

    epoch_nums = []
    training_loss = []
    validation_loss = []

    epochs = 100
    print('Training on', device)
    score = 0
    count = 0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, epoch)
        test_loss, correct_predictions = test(model, device, test_loader)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)

        if correct_predictions > score:
            torch.save(model.state_dict(), "/Users/Corty/Sync/becode_projects/Python/signature-recognition/model/cnn_2.pt")
            score = correct_predictions + 15
            count = 0

        elif count > 15:
            break

        count += 1

    return epoch_nums, training_loss, validation_loss


def try_model(file_name, model_path="/Users/Corty/Sync/becode_projects/Python/signature-recognition/model/cnn.pt"):
    from PIL import Image
    model = Net()
    file_path = "/Users/Corty/Sync/becode_projects/Python/signature-recognition/parsed_documents_CNN/batch/" + file_name + ".png"
    params = torch.load(model_path)
    model.load_state_dict(params)
    model.eval()
    transformation = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor()])
    img = Image.open(file_path)
    img_tensor = transformation(img).unsqueeze_(0)
    return model(img_tensor)


if __name__ == "__main__":
    epoch_nums, training_loss, validation_loss = main()
    plt.figure(figsize=(15, 15))
    plt.plot(epoch_nums, training_loss)
    plt.plot(epoch_nums, validation_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()

# import os
# import natsort
# for root, folders, files in os.walk("../parsed_documents_CNN/"):
#     for idx, file_name in enumerate(natsort.natsorted(files)):
#         if file_name != ".DS_Store":
#             print(f"{file_name[:-4]} : {try_model(file_name[:-4])[0][0] < try_model(file_name[:-4])[0][1]}")