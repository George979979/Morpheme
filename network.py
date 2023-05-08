import torch
import torch.nn as nn
from torch.nn import NLLLoss

from layers import Conv1D


class BasicNeuralTagger(nn.Module):

    def __init__(self, labels_number,
                 vocab_size=None, task_data=None,
                 device="cpu", clip=5.0, lr=1e-3, l2=0.0, min_prob=1.0,
                 modules=None, **kwargs):
        super(BasicNeuralTagger, self).__init__()
        self.vocab_size = vocab_size
        self.labels_number = labels_number
        self.task_data = task_data or dict()
        self.build_network(labels_number, input_dim=vocab_size, **kwargs)
        self.criterion = nn.NLLLoss(reduction="none")
        self.device = device
        self.clip = clip
        self.l2 = l2
        self.min_prob = torch.Tensor([min_prob])[0].to(device)
        if self.device is not None:
            self.to(self.device)
        modules = modules or dict()
        for key, module in modules.items():
            self.add_module(key, module)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def build_network(self, **kwargs):
        raise NotImplementedError("You should implement network construction in your derived class.")

    def forward(self, inputs):
        raise NotImplementedError("You should implement forward pass in your derived class.")

    def train_on_batch(self, x, y=None, task=None, mask=None):
        self.train()
        self.optimizer.zero_grad()
        curr_task_data = self.task_data.get(task, dict())
        loss_func = curr_task_data.get("func", BasicNeuralTagger._validate)
        loss = loss_func(self, x, y=y, mask=mask)
        loss_key = curr_task_data.get("loss_key", "loss")
        loss_weight = curr_task_data.get("weight", 1.0)
        task_loss = loss[loss_key] * loss_weight
        if hasattr(self, "l2") and self.l2 > 0.0:
            dense_layer = self.dense if hasattr(self, "dense") else self.morpheme_dense
            dense_params = torch.cat([x.view(-1) for x in dense_layer.parameters()], dim=-1)
            reg_loss = self.l2 * torch.norm(dense_params) ** 2
            loss["l2_loss"] = reg_loss
            task_loss += reg_loss
        task_loss.backward()
        if self.clip:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
        self.optimizer.step()
        return loss

    def validate_on_batch(self, x, y, task=None, mask=None):
        self.eval()
        with torch.no_grad():
            return self._validate(x, y, mask=mask)

    def _validate(self, x, y, mask=None, task=None):
        if self.device is not None:
            y = y.to(self.device)
        log_probs = self(**x)
        if y.dim() > 1:
            permute_mask = (0, y.dim()) + tuple(range(1, y.dim()))
            loss = self.criterion(log_probs.permute(permute_mask), y)
        else:
            loss = self.criterion(log_probs, y)
        if mask is not None:
            loss = loss * mask
        loss = nn.functional.relu(loss + torch.log(self.min_prob)).mean()
        _, labels = torch.max(log_probs, dim=-1)
        return {"loss": loss, "labels": labels}


class BertMorphemeLetterModel(BasicNeuralTagger):

    def __init__(self, labels_number, vocab_size=None,
                 task_data=None,
                 device="cpu", clip=5.0,
                 lr=1e-3, bert_lr_mult=0.1, l2=0.0,
                 weight_decay=0.0, min_prob=1.0,
                 **kwargs):
        super(BasicNeuralTagger, self).__init__()
        self.vocab_size = vocab_size
        self.labels_number = labels_number
        self.task_data = task_data or dict()
        print(kwargs)
        self.device = device
        self.build_network(labels_number, **kwargs, input_dim=vocab_size)
        self.criterion = nn.NLLLoss(reduction="none")
        self.clip = clip
        self.l2 = l2
        self.min_prob = torch.Tensor([min_prob])[0].to(device)
        if self.device is not None:
            self.to(self.device)
        if bert_lr_mult != 1.0:
            bert_params = [value for key, value in self.named_parameters() if
                           key in ["projection.weight", "projection.bias"]]
            other_params = [value for key, value in self.named_parameters() if
                            key not in ["projection.weight", "projection.bias"]]
            self.optimizer = torch.optim.Adam(
                [
                    {'params': bert_params, 'lr': lr * bert_lr_mult},
                    {'params': other_params}
                ],
                lr=lr,
                weight_decay=weight_decay,
                eps=1e-7
            )
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _build_basic_network(self, letters_number,
                             input_dim=None, n_letter_embedding=32,
                             use_bert=True, use_subtoken_weights=False,
                             subtoken_vocab_size=None,
                             d=10, n_proj=256,
                             n_layers=1, window=5, n_hidden=256,
                             use_embedding_relu=False,
                             use_batch_norm=True,
                             dropout=0.1, n_before_dense=None,
                             **kwargs
                             ):
        self.letters_number = letters_number
        self.n_letter_embedding = n_letter_embedding
        self.use_bert = use_bert
        self.use_subtoken_weights = use_subtoken_weights
        self.d = d
        self.n_proj = n_proj
        self.n_layers = n_layers
        self.window = window
        self.n_hidden = n_hidden
        self.use_embedding_relu = use_embedding_relu
        self.use_batch_norm = use_batch_norm
        # layers
        self.dropout = torch.nn.Dropout(dropout)
        if self.use_bert:
            if self.use_subtoken_weights:
                assert subtoken_vocab_size is not None
                self.subtoken_weights = torch.nn.Parameter(
                    data=torch.Tensor(subtoken_vocab_size), requires_grad=True
                )
            self.projection = torch.nn.Linear(d * input_dim, n_proj)
        if self.n_letter_embedding is not None:
            self.letter_embedding = torch.nn.Embedding(letters_number, self.n_letter_embedding)
        if self.use_embedding_relu:
            self.embedding_relu = torch.nn.ReLU()
        self.conv_layer = Conv1D(
            self.hidden_dim, n_layers, window, n_hidden, dropout=dropout, use_batch_norm=use_batch_norm
        )

    def build_network(self, labels_number, letters_number,
                      input_dim=None, n_letter_embedding=32,
                      use_bert=True, d=10, n_proj=256,
                      n_layers=1, window=5, n_hidden=256,
                      use_embedding_relu=False, use_batch_norm=True,
                      dropout=0.1, n_before_dense=None,
                      **kwargs):
        self._build_basic_network(
            letters_number, input_dim=input_dim,
            n_letter_embedding=n_letter_embedding,
            use_bert=use_bert, d=d, n_proj=n_proj,
            n_layers=n_layers, window=window,
            n_hidden=n_hidden,
            use_embedding_relu=use_embedding_relu,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            **kwargs
        )
        self.n_before_dense = n_before_dense
        last_n_hidden = self.conv_layer.output_dim if self.n_layers > 0 else self.n_hidden
        if self.n_before_dense is not None:
            self.before_dense = nn.Linear(last_n_hidden, self.n_before_dense)
            last_n_hidden = self.n_before_dense
        self.dense = nn.Linear(last_n_hidden, labels_number)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def _basic_forward(self, letters, **kwargs):
        inputs = kwargs.get("inputs") if self.use_bert else None
        subtoken_indexes = kwargs.get("subtoken_indexes") if self.use_bert else None
        if self.n_letter_embedding is not None:
            letter_embeddings = self.letter_embedding(letters)
        else:
            letter_embeddings = torch.nn.functional.one_hot(letters, self.letters_number).float()
        if inputs is not None:
            if self.use_subtoken_weights and subtoken_indexes is not None:
                input_weights = torch.unsqueeze(torch.sigmoid(self.subtoken_weights[subtoken_indexes]), dim=-1)
                inputs *= input_weights
            shape = inputs.shape
            new_shape = tuple(shape[:-2]) + (shape[-2] * shape[-1],)
            inputs = inputs.reshape(new_shape)
            hidden_outputs = self.projection(inputs)
            hidden_outputs = torch.cat([hidden_outputs, letter_embeddings], dim=-1)
        else:
            hidden_outputs = letter_embeddings
        if self.use_embedding_relu:
            hidden_outputs = self.embedding_relu(hidden_outputs)
        if self.n_layers > 0:
            conv_outputs = self.conv_layer(hidden_outputs)
        else:
            conv_outputs = hidden_outputs
        return conv_outputs

    def forward(self, letters, **kwargs):
        conv_outputs = self._basic_forward(letters, **kwargs)
        if self.n_before_dense is not None:
            conv_outputs = self.before_dense(conv_outputs)
        logits = self.dense(conv_outputs)
        log_probs = self.log_softmax(logits)
        return log_probs

    @property
    def hidden_dim(self):
        letter_embedding_size = self.n_letter_embedding or self.letters_number
        return self.n_proj * int(self.use_bert) + letter_embedding_size

