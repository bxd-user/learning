import torch
from torch import nn
from d2l import torch as d2l
import torchvision
from torch.utils import data
from torchvision import transforms

class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)
    
net = nn.Sequential(
    Reshape(),nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Flatten(),
    nn.Linear(16*5*5,120),nn.Sigmoid(),
    nn.Linear(120,84),nn.Sigmoid(),
    nn.Linear(84,10)
)

def load_data_mnist(batch_size):
    trans=transforms.ToTensor()
    mnist_train=torchvision.datasets.MNIST(root="./data",train=True,transform=trans,download=True)
    mnist_test=torchvision.datasets.MNIST(root="./data",train=False,transform=trans,download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True),
            data.DataLoader(mnist_test, batch_size, shuffle=False))

batch_size=256
train_iter,test_iter=load_data_mnist(batch_size)

def evaluate_accuracy_gpu(net,data_iter,device=None):
    if device is None and isinstance(net,torch.nn.Module):
        device=list(net.parameters())[0].device
    net.eval()
    total_correct=0.0
    total_samples=0.0

    with torch.no_grad():
        for X,y in data_iter:
            if isinstance(X,tuple):
                X=[x.to(device) for x in X]
            else:
                X=X.to(device)
            y=y.to(device)
            y_hat=net(X)
            batch_correct=accuracy(y_hat,y)
            batch_samples=y.numel()
            
            total_correct+=batch_correct
            total_samples+=batch_samples

    return total_correct/total_samples

def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat=y_hat.argmax(axis=1)
    cmp=y_hat.type(y.dtype)==y
    return float(cmp.sum())

def train_epoch_gpu(net,train_iter,loss,updater,device):
    net.train()
    total_loss=0.0
    total_correct=0.0
    total_samples=0.0

    for X,y in train_iter:
        if isinstance(X,tuple):
            X=[x.to(device) for x in X]
        else:
            X=X.to(device)
        y=y.to(device)
        y_hat=net(X)
        l=loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
            batch_loss = float(l.mean().detach()) * len(y)
            batch_correct = accuracy(y_hat, y)
            batch_samples = y.numel()
            
            total_loss += batch_loss
            total_correct += batch_correct
            total_samples += batch_samples
        else:
            l.sum().backward()
            updater(X.shape[0])
            batch_loss = float(l.sum().detach())
            batch_correct = accuracy(y_hat, y)
            batch_samples = y.numel()
            
            total_loss += batch_loss
            total_correct += batch_correct
            total_samples += batch_samples
            
    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples
    return avg_loss, avg_accuracy

lr=5
num_epochs=10

def train_mnist(net,train_iter,test_iter,loss,num_epochs,updater,device=d2l.try_gpu()):
    for epoch in range(num_epochs):
        train_loss,train_acc=train_epoch_gpu(net,train_iter,loss,updater,device)
        test_acc=evaluate_accuracy_gpu(net,test_iter,device)
        print(f'epoch {epoch+1}, loss {train_loss:.4f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')

if __name__=='__main__':
    net=net.to(d2l.try_gpu())
    loss=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(net.parameters(),lr=lr)
    train_mnist(net,train_iter,test_iter,loss,num_epochs,optimizer)