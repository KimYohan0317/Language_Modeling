# import some packages you need here
import torch
from model_ import CharRNN, CharLSTM
from dataset import Shakespeare
import torch.nn.functional as F

def generate(model_name, seed_characters, temperature, max_length, dataset):
    """ Generate characters

    Args:
        model_name: 'RNN' or 'LSTM'
        seed_characters: seed characters
        temperature: T
        max_length: maximum length of generated text
        dataset: instance of Shakespeare dataset

    Returns:
        samples: generated characters
    """

    # Initialize model parameters
    input_size = len(dataset.char_to_idx)
    output_size = len(dataset.char_to_idx)
    hidden_size = 128
    num_layers = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model based on the model_name
    if model_name == 'RNN':
        model = CharRNN(input_size, hidden_size, output_size, num_layers=num_layers).to(device)
        model_path = './best_RNN_model.pt'
    elif model_name == 'LSTM':
        model = CharLSTM(input_size, hidden_size, output_size, num_layers=num_layers).to(device)
        model_path = './best_LSTM_model.pt'
        
    # Load the trained model
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Initialize the hidden state
    hidden = model.init_hidden(1)
    with torch.no_grad():
        samples = seed_characters
        
        # Preprocess seed_characters as per the dataset preprocessing
        seed_characters = seed_characters.lower().strip()
        seed_encoded = [dataset.char_to_idx[char] for char in seed_characters]
        
        #one-hot encoding
        seed_input = torch.zeros(1, len(seed_encoded), input_size)
        for i, idx in enumerate(seed_encoded):
            seed_input[0, i, idx] = 1.0
        seed_input = seed_input.to(device)
        
        # Generate characters
        for _ in range(max_length):
            output, hidden = model(seed_input, hidden)
            
            # Sample from the network as a multinomial distribution
            output_dist = output.view(-1).div(temperature).exp()
            
            # Ensure the multinomial sampling respects the valid range of indices
            while True:
                top_i = torch.multinomial(output_dist, 1)[0]
                if top_i.item() in dataset.idx_to_char:
                    break
            
            # Add predicted character to the samples and use as next input
            predicted_char = dataset.idx_to_char[top_i.item()]
            samples += predicted_char
            
            # Use predicted character as next input (one-hot encoded)
            seed_input.fill_(0)
            seed_input[0, 0, top_i] = 1.0
        
    return samples

if __name__ == '__main__':
    input_file = './shakespeare_train.txt'
    dataset = Shakespeare(input_file)
    
    model_name = 'RNN'
    temperature = 1
    max_length = 100
    seed_characters = 'We are'
    
    generated_text = generate(model_name, seed_characters, temperature, max_length, dataset)
    print(generated_text)
