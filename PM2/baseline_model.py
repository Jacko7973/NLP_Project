import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable
import random
import time
from human_eval.data import write_jsonl, read_problems

from ply_tokenizer import Vocabulary, tokenize, untokenize
from data_loader import load_jsonl

LOAD_PRETRAINED = True


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda') # Nvida hardware acceleration
elif torch.backends.mps.is_available():
    device = torch.device('mps') # MacOS hardware acceleration

class LSTMModel(nn.Module):
    def __init__(self, dims):
        super().__init__()

        self.vocab = Vocabulary.load("vocab.json")
        self.dims = dims

        self.w = nn.Linear(in_features=len(self.vocab), out_features=self.dims, device=device)
        self.u_f = nn.Linear(in_features=self.dims, out_features=self.dims, device=device)
        self.w_f = nn.Linear(in_features=self.dims, out_features=self.dims, device=device)
        self.u_i = nn.Linear(in_features=self.dims, out_features=self.dims, device=device)
        self.w_i = nn.Linear(in_features=self.dims, out_features=self.dims, device=device)
        self.u_c = nn.Linear(in_features=self.dims, out_features=self.dims, device=device)
        self.w_c = nn.Linear(in_features=self.dims, out_features=self.dims, device=device)
        self.u_o = nn.Linear(in_features=self.dims, out_features=self.dims, device=device)
        self.w_o = nn.Linear(in_features=self.dims, out_features=self.dims, device=device)
        self.v = nn.Linear(in_features=self.dims, out_features=len(self.vocab), device=device)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
    def start(self):
        h = torch.zeros(self.dims, device=device)
        c = torch.zeros(self.dims, device=device)
        return (h, c)

    def step(self, state, idx):
        """    Pass idx through the layers of the model.
            Return a tuple containing the updated hidden state (h) and cell state (c), and the log probabilities of the predicted next character."""
        h, c = state
        x = F.one_hot(torch.tensor(idx), num_classes=len(self.vocab)).float().to(device)
        x_new = self.w(x)
        f_t = self.sigmoid(self.u_f(h) + self.w_f(x_new))
        i_t = self.sigmoid(self.u_i(h) + self.w_i(x_new))
        c_new = self.tanh(self.u_c(h) + self.w_c(x_new))
        c_t = f_t * c + i_t * c_new
        o_t = self.sigmoid(self.u_o(h) + self.w_o(x_new))
        h_t = o_t * self.tanh(c_t)
        y_t = self.log_softmax(self.v(h_t))
        return (h_t, c_t), y_t

    def predict(self, state, idx):
        """    Obtain the updated state and log probabilities after calling self.step().
            Return the updated state and the most likely next symbol."""
        new_state, log_probs = self.step(state, idx)
        symbol = self.vocab.get_token(torch.argmax(log_probs))
        return new_state, symbol

    def fit(self, data: Iterable[dict], val_data: Iterable[dict] = None, lr=0.001, epochs=10):
        """    This function is identical to fit() from part2.py.
            The only exception: the state to keep track is now the tuple (h, c) rather than just h. This means after initializing the state with the start state, detatch it from the previous computattion graph like this: `(state[0].detach(), state[1].detach())`"""

        optim = torch.optim.Adam(self.parameters(), lr)
        lossfn = nn.NLLLoss()

        for epoch in range(epochs):
            start_time = time.time()
            self.train(True)
            random.shuffle(data)
            total_loss = 0
            total_chars = 0

            for item in data:
                sentence = '"""' + item["text"] + '"""' + "\n\n" + item["code"]
                sentence_tokens = tokenize(sentence)
                # print(sentence)

                state = self.start()
                state = (state[0].detach(), state[1].detach())
                optim.zero_grad()
                sentence_loss = 0

                for i in range(1, len(sentence_tokens)):
                    prev, curr = sentence_tokens[i-1:i+1]
                    state, yt = self.step(state, prev)
                    loss = lossfn(yt.view(1, -1), torch.tensor(curr, device=device).view(1))
                    sentence_loss += loss
                    total_chars += 1

                sentence_loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                optim.step()
                total_loss += sentence_loss.item()

            avg_loss = total_loss / total_chars
            end_time = time.time()

            # Evaluate on validation set if provided
            val_acc_str = ""
            if val_data is not None:
                val_acc = self.evaluate(val_data)
                val_acc_str = f' Val acc: {val_acc:.4f}'

            print(f'Epoch: {epoch+1:-6} Average loss: {avg_loss:-6.2f}{val_acc_str} ({end_time-start_time:.2f}s)')

    def evaluate(self, data):
        """    Iterating over the sentences in the data, calculate next character prediction accuracy.
            Use `self.eval()` and `with torch.no_grad()` so that the model is not trained during evaluation.
            Use self.predict() to get the predicted next character, and then check if it matches the real next character found in the data.
            Divide the total correct predictions by the total number of characters to get the final accuracy.
            The code may be identitcal to evaluate() from part2.py."""
        self.eval()

        total_chars = 0
        total_correct = 0
        with torch.no_grad():
            for item in data:
                sentence = '"""' + item["text"] + '"""' + "\n\n" + item["code"]
                sentence_tokens = tokenize(sentence)

                state = self.start()
                for i in range(1, len(sentence_tokens)):
                    prev, curr = sentence_tokens[i-1:i+1]
                    state, pred_symbol = self.predict(state, prev)
                    target_symbol = self.vocab.get_token(curr)

                    total_chars += 1
                    total_correct += 1 if pred_symbol == target_symbol else 0

        return total_correct / total_chars

if __name__ == '__main__':
    
    print("[INFO] Loading data...")
    data = list(load_jsonl("mbpp.jsonl"))
    random.shuffle(data)
    split_train = math.floor(len(data) * 0.7)
    split_val = math.floor(len(data) * 0.8)

    train_data = data[:split_train]
    val_data = data[split_train:split_val]
    test_data = data[split_val:]

    # DEBUG
    # print('[DEBUG] CLIPPING TRAIN DATA')
    # train_data = train_data[:5]

    model = LSTMModel(dims=128).to(device)
    if LOAD_PRETRAINED:
        print('[INFO] Loading pretrained weights from "lstm_model.pth"')
        checkpoint = torch.load('lstm_model.pth', map_location=device, weights_only=False)
        dims = checkpoint['dims']
        model = LSTMModel(dims).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    else:
        model.fit(train_data, val_data=val_data, epochs=10)
        print('[INFO] Saving model weights to "lstm_model.pth"')
        torch.save({
            'model_state_dict': model.state_dict(),
            'dims': model.dims
        }, 'lstm_model.pth')

    """Use this code if you saved the model and want to load it up again to evaluate. Comment out the model.fit() and torch.save() code if so.
    # checkpoint = torch.load('rnn_model.pth', map_location=device, weights_only=False)
    # vocab = checkpoint['vocab']
    # dims = checkpoint['dims']
    # model = RNNModel(vocab, dims).to(device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    """
    breakpoint()

    model.eval()
    print(model.evaluate(val_data), model.evaluate(test_data))

    def generate_one_completion(text):
        prompt = '"""' + text + '"""' + "\n\n"
        prompt_tokens = tokenize(prompt)

        tokens_generated = []
        with torch.no_grad():
            state = model.start()
            for i in range(1, len(prompt_tokens)-1):
                prev, _ = prompt_tokens[i-1:i+1]
                state, _ = model.predict(state, prev)

            prev_tok = prompt_tokens[-1]
            while len(tokens_generated) < 100:
                state, next_tok = model.predict(state, model.vocab.get_id(prev_tok))
                tokens_generated.append(next_tok)
                if next_tok == model.vocab.get_id("<EOS>"):
                    break

                prev_tok = next_tok
        
        return untokenize(tokens_generated)

    print("Running HumanEval benchmark problems")
    problems = read_problems()

    num_samples_per_task = 200
    samples = [
        dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
        for task_id in problems
        for _ in range(num_samples_per_task)
    ]
    write_jsonl("samples.jsonl", samples)
