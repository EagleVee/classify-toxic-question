import numpy as np
import torch
import torch.nn as nn


class InsincereModel(nn.Module):
    def __init__(self, device, hidden_dim, hidden_dim_fc, embedding_matrixs, vocab_size=None, embedding_dim=None,
                 dropout=0.1, num_capsule=5, dim_capsule=5, capsule_out_dim=1, alpha=0.8, beta=0.8,
                 finetuning_vocab_size=120002,
                 embedding_mode='mixup', max_seq_len=70):
        super(InsincereModel, self).__init__()
        self.beta = beta
        self.embedding_mode = embedding_mode
        self.finetuning_vocab_size = finetuning_vocab_size
        self.alpha = alpha
        vocab_size, embedding_dim = embedding_matrixs[0].shape
        self.raw_embedding_weights = embedding_matrixs
        self.embedding_0 = nn.Embedding(vocab_size, embedding_dim, padding_idx=0).from_pretrained(
            torch.from_numpy(embedding_matrixs[0]))
        self.embedding_1 = nn.Embedding(vocab_size, embedding_dim, padding_idx=0).from_pretrained(
            torch.from_numpy(embedding_matrixs[1]))
        self.embedding_mean = nn.Embedding(vocab_size, embedding_dim, padding_idx=0).from_pretrained(
            torch.from_numpy((embedding_matrixs[0] + embedding_matrixs[1]) / 2))
        self.learnable_embedding = nn.Embedding(finetuning_vocab_size, embedding_dim, padding_idx=0)
        nn.init.constant_(self.learnable_embedding.weight, 0)
        self.learn_embedding = False
        self.spatial_dropout = nn.Dropout2d(p=0.2)
        self.device = device
        self.hidden_dim = hidden_dim
        self.rnn0 = nn.LSTM(embedding_dim, int(hidden_dim / 2), num_layers=1, bidirectional=True, batch_first=True)
        self.rnn1 = nn.GRU(hidden_dim, int(hidden_dim / 2), num_layers=1, bidirectional=True, batch_first=True)
        self.capsule = Capsule(input_dim_capsule=self.hidden_dim, num_capsule=num_capsule, dim_capsule=dim_capsule)
        self.dropout2 = nn.Dropout(0.3)
        self.lincaps = nn.Linear(num_capsule * dim_capsule, capsule_out_dim)
        self.attention1 = Attention(self.hidden_dim, max_seq_len=max_seq_len)
        self.attention2 = Attention(self.hidden_dim, max_seq_len=max_seq_len)
        self.fc = nn.Linear(hidden_dim * 4 + capsule_out_dim, hidden_dim_fc)
        self.norm = torch.nn.LayerNorm(hidden_dim * 4 + capsule_out_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim_fc, 1)

    def set_embedding_mode(self, embedding_mode):
        self.embedding_mode = embedding_mode

    def enable_learning_embedding(self):
        self.learn_embedding = True

    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for k in ih:
            nn.init.xavier_uniform_(k)
        for k in hh:
            nn.init.orthogonal_(k)
        for k in b:
            nn.init.constant_(k, 0)

    def apply_spatial_dropout(self, emb):
        emb = emb.permute(0, 2, 1).unsqueeze(-1)
        emb = self.spatial_dropout(emb).squeeze(-1).permute(0, 2, 1)
        return emb

    def forward(self, seqs, lens, return_logits=True):
        # forward embeddings
        if self.embedding_mode == 'mixup':
            emb0 = self.embedding_0(seqs)  # batch_size x seq_len x embedding_dim
            emb1 = self.embedding_1(seqs)
            prob = np.random.beta(self.alpha, self.beta, size=(seqs.size(0), 1, 1)).astype(np.float32)
            prob = torch.from_numpy(prob).to(self.device)
            emb = emb0 * prob + emb1 * (1 - prob)
        elif self.embedding_mode == 'emb0':
            emb = self.embedding_0(seqs)
        elif self.embedding_mode == 'emb1':
            emb = self.embedding_1(seqs)
        elif self.embedding_mode == 'mean':
            emb = self.embedding_mean(seqs)
        else:
            assert False
        if self.learn_embedding:
            seq_clamped = torch.clamp(seqs, 0, self.finetuning_vocab_size - 1)
            emb_learned = self.learnable_embedding(seq_clamped)
            emb = emb + emb_learned
        emb = self.apply_spatial_dropout(emb)
        # forward rnn encoder
        lstm_output0, _ = self.rnn0(emb)
        lstm_output1, _ = self.rnn1(lstm_output0)
        # forward capsule
        content3 = self.capsule(lstm_output1)
        batch_size = content3.size(0)
        content3 = content3.view(batch_size, -1)
        content3 = self.dropout2(content3)
        content3 = torch.relu(self.lincaps(content3))
        # forward feature extractor
        feature_att1 = self.attention1(lstm_output0)
        feature_att2 = self.attention2(lstm_output1)
        feature_avg2 = torch.mean(lstm_output1, dim=1)
        feature_max2, _ = torch.max(lstm_output1, dim=1)
        feature = torch.cat((feature_att1, feature_att2, feature_avg2, feature_max2, content3), dim=-1)
        feature = self.norm(feature)
        feature = self.dropout1(feature)
        feature = torch.relu(feature)
        # forward dense layer
        out = self.fc(feature)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)  # batch_size x 1
        if not return_logits:
            out = torch.sigmoid(out)
        return out
