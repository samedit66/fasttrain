# fasttrain
With `fasttrain` you'll forever forget about ugly and complex training loops in PyTorch!

To start with, let's create a simple convnet just from the PyTorch tutorial:
```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

learning_rate = 1e-3
batch_size = 64
epochs = 5

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
```

Your next move will probably be building some kind of training and testing functions to, of course, train your model and show how effective it is, but let's forget about it, and use little help from the `Trainer` class:
```python
from fasttrain.train.trainer import Trainer
from fasttrain.metrics import accuracy


class FashionMNISTTrainer(Trainer):

    def predict(self, input_batch):
        (x_batch, _) = input_batch
        return self.model(x_batch)

    def compute_loss(self, input_batch, output_batch):
        (_, y_batch) = input_batch
        return nn.CrossEntropyLoss()(output_batch, y_batch)

    def eval_metrics(self, input_batch, output_batch):
        (_, y_batch) = input_batch
        return {
            "accuracy": accuracy(output_batch, y_batch, task="multiclass")
        }
```
With `Trainer` all you have to do is specify how you predictions are made, how to compute loss and how to evaluate metrics (I hope you've seen that I've also imported `accuracy` metric, isn't it just fancy?). The rest you have to do is specify the model optimizer and call the `train` function:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
trainer = FashionMNISTTrainer(model, optimizer)
history = trainer.train(train_dataloader, val_data=test_dataloader, num_epochs=epochs, callbacks=[Tqdm(colab=True)])
```
`fasttrain` comes with batteries and offers some useful "callbacks" - one of them is `Tqdm` which shows a pretty-looking progress bar (`colab=True` option is used 'cause I build this network in Google Colab, if you're using it locally you don't need to specify it, only when in Colab). Let's see how it looks:
![training_loop](https://github.com/samedit66/fasttrain/assets/45196253/edecaee0-1c92-4a9f-ac3d-639c458a2ab5)

Did you see it? The first line printed tells us that we're using cuda - we never mentioned that, did we? `Trainer` is smart enough to use cuda if it's enabled, but if you want you can specify device which you want to use in `train()` with, for example, option `device='cpu'`. `train()` also returns us the history of training. What is it? It contains kind of dict which by key returns metrics' statistics over epochs. So you can later use matplotlib to show them. But `fasstrain` has a better option: plot them right now!
```python
history.plot("loss", with_val=True)
```
![loss](https://github.com/samedit66/fasttrain/assets/45196253/efc0c9e9-4459-4bce-81ec-3c1a53cf51f1)
```python
history.plot("accuracy", with_val=True)
```
![accuracy](https://github.com/samedit66/fasttrain/assets/45196253/336bdef0-9f06-4887-8cb5-05255c89b228)

Pretty-looking metrics with graphs, remember, batteries ARE included!
