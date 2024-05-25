## Language_Modeling

### Shakespeare dataset "many-to-many" character-level language modeling task

---

### file description

- `dataset.py`: Contains functions to load and preprocess the dataset.
- `model.py`: Defines the RNN and LSTM models.
- `main.py`: Implements the training and validation loops, saves the trained models (`model.pt`), and plots the training curves (`save_plot`).
- `generate.py`: Contains functions to generate text using the trained models.
- `shakespeare_train.txt`: Dataset file containing the Shakespeare text used for training.
- `best_LSTM_model.pt`: Saved weights of the best LSTM model.
- `best_RNN_model.pt`: Saved weights of the best RNN model.

---

### Model Description

#### RNN

- Simple RNN model used for character-level language modeling of Shakespeare dataset.

#### LSTM

- Long Short-Term Memory (LSTM) model used for character-level language modeling of Shakespeare dataset.

##### Implementation Details:

- `CharRNN` Class:
  - **Architecture:**
    - Input: One-hot encoded character vectors.
    - RNN Layer: Single-layer RNN with `hidden_size` units.
    - Fully Connected Layer: Maps the RNN output to the output size.
  - **Initialization:**
    - Initializes the hidden state with zeros.
    - Uses `nn.RNN` module for the recurrent layer.

- `CharLSTM` Class:
  - **Architecture:**
    - Input: One-hot encoded character vectors.
    - LSTM Layer: Single-layer LSTM with `hidden_size` units.
    - Fully Connected Layer: Maps the LSTM output to the output size.
  - **Initialization:**
    - Initializes the hidden and cell states with zeros.
    - Uses `nn.LSTM` module for the recurrent layer.

Both models (`CharRNN` and `CharLSTM`) are capable of learning dependencies in sequential data and are trained on the Shakespeare dataset for language modeling tasks.

---

## Experiment result
#### RNN Model

- **RNN train & valid Loss:**
  
  ![RNN train & valid Loss](https://github.com/KimYohan0317/Language_Modeling/blob/main/train_RNN.png)

#### LSTM Model

- **LSTM train & valid Loss:**
  
  ![LSTM train & valid Loss](https://github.com/KimYohan0317/Language_Modeling/blob/main/train_LSTM.png)

---

#### train & valid Loss

| Model | train Loss | valid loss |
|-------|-----------|-----------|
|   RNN   |    1.4822  |    1.4875  |
|   LSTM   |    1.2504  |    1.2707  |

---

### best model
- LSTM is superior to RNN in terms of validation loss.

### Exploring Text Generation: Sampling 5 Different Text Samples of at Least 100 Characters from Seed Characters
- Seed characters: we / you / I / Would / If
- Results
  - (1) **we** - weeplerin an talerys,werinores falinedrers;werinores, hop talitoutoronery, ighounely.cyino, yrely-boli
  - (2) **you** - youonedronabrines:as y torurero tounedes,we fes ines isimineraranedinenes yedededorererineneranenes'dul
  - (3) **I** - Inistle consul?thou maybroicest which thy defeclicy, dorset me harm,how can i have to kill in coriola
  - (4) **Would** - Wouldledinedineneranorininanononerinesi'dinesivimaronoranoforinesbalinonenanesberinesininedinoresinoneded
  - (5) **If** - Iflyon, igh, aver talerorys,whitous gre, ighed;ak on,ared hosyoronedinedanededary my;arinelyinoredetse

### Experimenting with Temperature in Text Generation
- Temperatures: 0.5 / 0.7 / 1.0 / 1.5 
- Results 
  - **0.5** - Ing such conclusion,because i cannot determined the world face the timein the prince, and then death.
  - **0.7** - Ith no laugh.menenius:that's of my soul; in the city,will deserved me justices browned cushions fair 
  - **1.0** - Inistle consul?thou maybroicest which thy defeclicy, dorset me harm,how can i have to kill in coriola
  - **1.5** - Istings gain: yourn heor'd?first murderer:owand with anus moved clarence in; which movocatise.mah? lo
 
### Discussion

The LSTM model outperformed the RNN in terms of validation loss, suggesting its superiority in this language modeling task. Text samples generated from different seed characters show the model's ability to mimic Shakespearean style. Lower temperature values (0.5 and 0.7) produce more coherent and structured text, while higher values (1.0 and 1.5) generate more diverse but sometimes less coherent outputs.

Further improvements could involve fine-tuning the model architecture, experimenting with different hyperparameters, and exploring techniques like beam search for text generation to potentially enhance the quality of generated text.

---
