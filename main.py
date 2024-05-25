import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import Shakespeare
from model_ import CharRNN, CharLSTM
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# import some packages you need here


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function
    
    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """

    # write your codes here
    model.train()
    total_loss = 0
    for input, target in tqdm(trn_loader):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        hidden = model.init_hidden(input.size(0))
        output, hidden = model(input, hidden)
        loss = criterion(output.view(-1, output.size(2)), target.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input.size(0)
    trn_loss = total_loss / len(trn_loader.dataset)
    return trn_loss

def validate(model, val_loader, device, criterion):
    # """ Validate function
    # Args:
    #     model: network
    #     val_loader: torch.utils.data.DataLoader instance for testing
    #     device: device for computing, cpu or gpu
    #     criterion: cost function

    # Returns:
    #     val_loss: average loss value
    # """

    # write your codes here
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input, target in tqdm(val_loader):
            input, target = input.to(device), target.to(device)
            hidden = model.init_hidden(input.size(0))  # Batch size passed here
            output, hidden = model(input, hidden)
            loss = criterion(output.view(-1, output.size(2)), target.view(-1))
            total_loss += loss.item() * input.size(0)
    val_loss = total_loss / len(val_loader.dataset)
    return val_loss


def main():
    # """ Main function

    #     Here, you should instantiate
    #     1) DataLoaders for training and validation. 
    #        Try SubsetRandomSampler to create these DataLoaders.
    #     3) model
    #     4) optimizer
    #     5) cost function: use torch.nn.CrossEntropyLoss

    # """

    # write your codes here
    model_name = 'RNN'
    batch_size = 256
    dataset = Shakespeare('shakespeare_train.txt')
    num_samples = len(dataset)
    num_epochs = 50
    hidden_size = 128
    num_layers = 1

    
    trn_size = int(0.8 * num_samples)
    val_size = num_samples - trn_size
    trn_dataset, val_dataset = random_split(dataset, [trn_size, val_size])
    
    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    
    input_size = len(dataset.char_to_idx)
    output_size = len(dataset.char_to_idx)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if model_name == 'RNN':
        model = CharRNN(input_size, hidden_size, output_size, num_layers=num_layers).to(device)
    elif model_name == 'LSTM':
        model = CharLSTM(input_size, hidden_size, output_size, num_layers=num_layers).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    
    trn_loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        trn_loss = train(model, trn_loader, device, criterion, optimizer)
        val_loss = validate(model, val_loader, device, criterion)
        print(f'Epoch [{epoch+1}/{num_epochs}], Trn Loss: {trn_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        trn_loss_list.append(trn_loss)
        val_loss_list.append(val_loss)
        
        #save the model with the best validation loss
        #save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_{}_model.pt'.format(model_name))
    #plotting best validation loss
    plt.figure(figsize=(8,6))
    sns.lineplot(trn_loss_list, marker='o', color='blue', label='training')
    sns.lineplot(val_loss_list, marker='o', color='orange', label='validation')

    plt.legend()
    plt.title(model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('train_{}.png'.format(model_name))
    plt.show()
if __name__ == '__main__':
    main()