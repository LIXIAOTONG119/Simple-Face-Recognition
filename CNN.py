import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, datasets
from PIL import Image

# Set Parameter
num_Batch = 8
Learning_rate = 0.01
Epoch = 21

# Load data
def readImg(path):
    return Image.open(path)


data_transform = transforms.Compose([
        transforms.ToTensor()
        ])#transfer the image to Tensor,nomalize to [0,1]
Train_Data = datasets.ImageFolder(root='CNN_PIE/TrainSet', transform=data_transform, loader=readImg)
Test_Data = datasets.ImageFolder(root='CNN_PIE/TestSet', transform=data_transform, loader=readImg)
train_loader = DataLoader(Train_Data, batch_size=num_Batch, shuffle=True, num_workers=0)
test_loader = DataLoader(Test_Data, batch_size=num_Batch, shuffle=True, num_workers=0)

# Specialize the CNN Net
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(
                                   in_channels=1,      # input height
                                   out_channels=20,    # n_filters
                                   kernel_size=5,      # filter size
                                   stride=1,           # filter movement/step
                                   padding=2,      # padding=(kernel_size-1)/2 when stride=1
                                   ),      # output shape 
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(
                                   in_channels=20,      # input height
                                   out_channels=50,    # n_filters
                                   kernel_size=5,      # filter size
                                   stride=1,           # filter movement/step
                                   padding=2,      #  padding=(kernel_size-1)/2 when stride=1
                                   ),      # output shape 
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))

        self.fc = nn.Sequential(nn.Linear(50 * 64, 500),
                                nn.ReLU(),
                                nn.Linear(500, 21))

    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


# Using GPU for training
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
cnn = CNN()
cnn.to(device)
print(cnn)

# Create optimizer & loss
optimizer = optim.SGD(cnn.parameters(), lr=Learning_rate, momentum=0.6)
loss_F = nn.CrossEntropyLoss()


# Trianing
def train(i):
    cnn.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = cnn(data)
        loss = loss_F(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch= {} , Batch= {} ,Loss= {:g} '.format(i, batch_idx, loss.item()))

@torch.no_grad()
# Test
def test():
    cnn.eval()
    correct = 0
    total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = cnn(data)
        predict = torch.argmax(output, 1)
        total += target.size(0)
        correct += (predict == target).sum().item()

    accuracy = correct / total
    print('\nTest Accuracy= {:.1%}'.format(accuracy))


for i in range(1, Epoch):
    train(i)
    test()

