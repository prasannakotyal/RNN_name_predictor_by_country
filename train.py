import torch
from data import *
from model import *
import random
import time
import math
from torch.autograd import Variable
import torch.nn as nn

n_hidden = 128
n_epochs = 100000
print_every = 5000
plot_every = 1000
learning_rate = 0.001

# Initialize the RNN model
rnn = RNN(n_letters, n_hidden, n_categories)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

# Track the lowest loss and save the model when it improves
best_loss = float('inf')  # Set initial best_loss to a very high value
best_model_path = './models/best_model.pt'  # Path where the best model will be saved

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) 
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

def randomChoice(l):
    if len(l) == 0:
        raise ValueError("Cannot choose from an empty list.")
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
    if len(all_categories) == 0:
        raise ValueError("all_categories is empty.")
    
    category = randomChoice(all_categories)
    
    if len(category_lines[category]) == 0:
        raise ValueError(f"No lines available for category {category}.")
    
    line = randomChoice(category_lines[category])
    
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    
    return category, line, category_tensor, line_tensor

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.item()


current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for epoch in range(1, n_epochs + 1):
    try:
        category, line, category_tensor, line_tensor = randomTrainingPair()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

        # Print epoch number, loss, name and guess
        if epoch % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

        # Check if this is the best loss so far, and save the model if it is
        if loss < best_loss:
            best_loss = loss
            torch.save(rnn, best_model_path)  # Save the best model

    except ValueError as e:
        print(f"Error: {e}")
        break

print(f"Training complete. Best model saved with loss: {best_loss:.4f}")
