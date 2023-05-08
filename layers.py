import torch
import torch.nn as nn


class Conv1D(nn.Module):
    
    def __init__(self, input_dim, n_layers=1, window=5, hidden=64, dropout=0.0, use_batch_norm=False):
        super().__init__()
        self.input_dim = input_dim
        if isinstance(window, int):
            window = [window]
        if isinstance(hidden, int):
            hidden = [hidden] * len(window)
        self.n_layers = n_layers
        self.window = window
        self.hidden = hidden
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.convolutions = nn.ModuleList()
        for _ in range(self.n_layers):
            convolutions = nn.ModuleList()
            for i, (curr_window, curr_hidden) in enumerate(zip(self.window, self.hidden)):
                layer = nn.Conv1d(input_dim, curr_hidden, curr_window, padding=(curr_window // 2))
                convolutions.append(layer)
            curr_layer = {"convolutions": convolutions, "dropout": nn.Dropout(dropout)}
            if self.use_batch_norm:
                curr_layer["batch_norm"] = nn.BatchNorm1d(self.output_dim)
            self.convolutions.append(nn.ModuleDict(curr_layer))
            input_dim = self.output_dim

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        for layer in self.convolutions:
            outputs = []
            for sublayer, curr_window in zip(layer["convolutions"], self.window):
                curr_output = sublayer(inputs)
                if curr_window % 2 == 0:
                    curr_output = curr_output[..., :-1]
                outputs.append(curr_output)
            outputs = torch.cat(outputs, dim=1)
            if self.use_batch_norm:
                outputs = layer["batch_norm"](outputs)
            outputs = torch.nn.ReLU()(outputs)
            outputs = layer["dropout"](outputs)
            inputs = outputs
        outputs = outputs.permute(0, 2, 1)
        return outputs

    @property
    def output_dim(self):
        return sum(self.hidden)
        
