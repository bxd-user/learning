import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

def load_data_mnist(batch_size):
    trans=transforms.ToTensor()
    mnist_train=torchvision.datasets.MNIST(root="./data",train=True,transform=trans,download=True)
    mnist_test=torchvision.datasets.MNIST(root="./data",train=False,transform=trans,download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True),
            data.DataLoader(mnist_test, batch_size, shuffle=False))

def softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(1,keepdim=True)
    return X_exp/partition

def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat=y_hat.argmax(axis=1)
    cmp=y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net,data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    
    total_correct = 0.0
    total_samples = 0.0

    for X,y in data_iter:
        batch_correct=accuracy(net(X),y)
        batch_samples=y.numel()
        
        total_correct+=batch_correct
        total_samples+=batch_samples

    return total_correct/total_samples

def train_epoch(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    
    total_loss=0.0
    total_correct = 0.0
    total_samples = 0.0

    for X,y in train_iter:
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

def train_mnist(net,train_iter,test_iter,loss,num_epochs,updater):
    for epoch in range(num_epochs):
        train_metrics=train_epoch(net,train_iter,loss,updater)
        test_acc=evaluate_accuracy(net,test_iter)
        train_loss,train_acc=train_metrics
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train loss:    {train_loss:.4f}')
        print(f'  Train accuracy: {train_acc:.3f}')
        print(f'  Test accuracy:  {test_acc:.3f}')

batch_size=256
train_iter,test_iter=load_data_mnist(batch_size)

num_inputs=784
num_outputs=10

W=torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b=torch.zeros(num_outputs,requires_grad=True)
    
lr=0.1
    
def net(X):
    return softmax(torch.matmul(X.reshape(-1,W.shape[0]),W)+b)
    
def updater(batch_size):
    return d2l.sgd([W,b],lr,batch_size)
    
num_epochs=5
train_mnist(net,train_iter,test_iter,cross_entropy,num_epochs,updater)