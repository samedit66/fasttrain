# fasttrain
`fasttrain` is a lightweight framework for building training loops for neural nets as fast as possible. It's designed to remove all boring details about making up training loops in [PyTorch](https://pytorch.org/), so you don't have to concentrate on how to pretty print a loss or metrics or bother about how to calculate them right.

## Installation
Stable version:
```
pip install fasttrain
```
Latest version:
```
git clone https://github.com/samedit66/fasttrain.git
cd fasttrain
pip install .
```

## How do we start?
Let's use a neural network to classify images in the FashionMNIST dataset:
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

Then we make up a trainer:
```python
from fasttrain import Trainer
from fasttrain.metrics import accuracy

class MyTrainer(Trainer):

    # Define how we compute the loss
    def compute_loss(self, input_batch, output_batch):
        (_, y_batch) = input_batch
        return nn.CrossEntropyLoss()(output_batch, y_batch)

    # Define how we compute metrics
    def eval_metrics(self, input_batch, output_batch):
        (_, y_batch) = input_batch
        return {
            "accuracy": accuracy(output_batch, y_batch)
        }
```

Finally, let's train our model:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
trainer = MyTrainer(model, optimizer)
history = trainer.train(train_dataloader, val_data=test_dataloader, num_epochs=epochs)
```
`fasttrain` offers some useful callbacks - one of them is `Tqdm` which shows a pretty-looking progress bar:
![training_loop](https://github.com/samedit66/fasttrain/assets/45196253/edecaee0-1c92-4a9f-ac3d-639c458a2ab5)

`Trainer.train()` returns the history of training - it contains a dict which stores metrics over epochs and can plot them:
```python
history.plot("loss", with_val=True)
```
![loss](https://github.com/samedit66/fasttrain/assets/45196253/efc0c9e9-4459-4bce-81ec-3c1a53cf51f1)
```python
history.plot("accuracy", with_val=True)
```
![accuracy](https://github.com/samedit66/fasttrain/assets/45196253/336bdef0-9f06-4887-8cb5-05255c89b228)
