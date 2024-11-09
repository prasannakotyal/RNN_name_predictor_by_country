from model import *
from data import *
import sys
import torch

# Load the trained model
rnn = torch.load('./models/best_model.pt')

# Function to evaluate a given line (name)
def evaluate(line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output

# predict function to return top N predictions with probabilities
def predict(line, n_predictions=3):
    # Convert the input line (name) to tensor
    line_tensor = lineToTensor(line).requires_grad_()

    # model output
    output = evaluate(line_tensor)
    
    # top N categories and their log probabilities
    topv, topi = output.data.topk(n_predictions, 1, True)

    predictions = []
    for i in range(n_predictions):
        value = topv[0][i]               # Keep this as Tensor
        category_index = topi[0][i].item()  # Get the index of the category
        category = all_categories[category_index]
        probability = torch.exp(value).item()  #probability scores
        predictions.append((category, probability))

    return predictions

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <name> [num_predictions]")
        sys.exit(1)

    name = sys.argv[1]
    n_predictions = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    predicted_countries = predict(name, n_predictions=n_predictions)
    print(f"\nTop {n_predictions} predictions for the name '{name}':")
    for i, (country, prob) in enumerate(predicted_countries):
        print(f"{i + 1}: {country} with probability {prob:.4f}")
