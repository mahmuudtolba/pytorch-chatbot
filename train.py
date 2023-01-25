import json
from nltk_utils import tokenize , stem , bag_of_words
import numpy as np

from torch.utils.data import Dataset , DataLoader
import torch
import torch.nn as nn

from model import MyNeuralNet

with open("intents.json" , "r") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Loop through each sentence 
for intent in intents["intents"]:
    tag = intent["tag"]
    # add to tag list
    tags.append(tag)

    for pattern in intent["patterns"]:
        # tokenize each word in the sentence
        tokens = tokenize(pattern)
        # add to our words
        all_words.extend(tokens)
        #add to xy pair
        xy.append((tokens , tag))

# Stem and lower each word
ignore_words = ['!', '?' , '.']
all_words = [stem(word) for word in all_words if word not in ignore_words]
# Remove duplicate and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence , tag) in xy:
    # x : bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence , all_words)
    X_train.append(bag)
    # y : label of the tag
    label = tags.index(tag)
    y_train.append(label)


# Hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size , output_size)

# Class dataset
class ChatDataset(Dataset):
    def __init__(self , x ,y):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self , index):
        return self.x[index] , self.y[index]

dataset = ChatDataset(X_train , y_train)
train_loader = DataLoader(dataset=dataset,
                batch_size= batch_size,
                shuffle=True,
                num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MyNeuralNet(input_size , hidden_size , output_size).to(device)

# Loss and optimizer
loss_func = nn.CrossEntropyLoss()
opti = torch.optim.Adam(model.parameters() , lr = learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words , labels) in train_loader:
        words = words.to(device)
        labels = labels.to(labels)

        opti.zero_grad()
        outputs = model(words)
        loss = loss_func(outputs , labels)
        # Backward and updating the weigths
        loss.backward()
        opti.step()

print(f'final loss : {loss.item():0.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = 'data.pth'
torch.save(data, FILE)
print(f'training complete. file saved to {FILE}')
