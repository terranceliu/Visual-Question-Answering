# ref: https://github.com/zcyang/imageqa-san/blob/master/src/san_att_lstm_twolayer_theano.py
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ImageEmbedding(nn.Module):
    def __init__(self, output_size=1024):
        super(ImageEmbedding, self).__init__()
        self.cnn = models.vgg16(pretrained=True).features
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(512, output_size),
            nn.Tanh())

    def forward(self, image):
        # N * 448 * 448 -> N * 512 * 14 * 14
        feature_map = self.cnn(image)
        # N * 512 * 14 * 14 -> N * 512 * 196 -> N * 196 * 512
        feature_map = feature_map.view(-1, 512, 196).transpose(1, 2)
        # N * 196 * 512 -> N * 196 * 1024
        image_embedding = self.fc(feature_map)
        return image_embedding


class QuesEmbedding(nn.Module):
    def __init__(self, input_size=500, hidden_size=1024, num_layers=1, batch_first=True):
        super(QuesEmbedding, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first)

    def forward(self, ques):
        # seq_len * N * 500 -> (1 * N * 1024, 1 * N * 1024)
        _, hx = self.lstm(ques)
        # (1 * N * 1024, 1 * N * 1024) -> 1 * N * 1024
        h, _ = hx
        ques_embedding = h[0]
        return ques_embedding


class Attention(nn.Module):
    def __init__(self, d=1024, k=512):
        super(Attention, self).__init__()
        self.ff_image = nn.Linear(d, k)
        self.ff_ques = nn.Linear(d, k)
        self.ff_attention = nn.Linear(k, 1)

    def forward(self, vi, vq):
        # N * 196 * 1024 -> N * 196 * 512
        hi = self.ff_image(vi)
        # N * 1024 -> N * 512 -> N * 1 * 512
        hq = self.ff_ques(vq).unsqueeze(dim=1)
        # N * 196 * 512
        ha = F.tanh(hi + hq)
        # N * 196 * 512 -> N * 196 * 1 -> N * 196
        ha = self.ff_attention(ha).squeeze(dim=2)
        pi = F.softmax(ha)
        # (N * 196 * 1, N * 196 * 1024) -> N * 1024
        vi_attended = (pi.unsqueeze(dim=2) * vi).sum(dim=1)
        u = vi_attended + vq
        return u


class SANModel(nn.Module):
    def __init__(self, vocab_size, word_emb_size=500, emb_size=1024, att_ff_size=512, output_size=1000, num_att_layers=1, num_mlp_layers=1):
        super(SANModel, self).__init__()
        self.image_channel = ImageEmbedding(output_size=emb_size)

        self.word_embeddings = nn.Embedding(vocab_size, word_emb_size, padding_idx=1)
        self.ques_channel = QuesEmbedding(word_emb_size, output_size=emb_size, num_layers=1, batch_first=False)

        self.san = nn.ModuleList([Attention(d=emb_size, k=att_ff_size)] * num_att_layers)

        self.mlp = nn.Sequential(
            nn.Linear(emb_size, output_size),
            nn.Dropout(p=0.5))

    def forward(self, images, questions):
        embeds = self.word_embeddings(questions)
        image_embeddings = self.image_channel(images)
        ques_embeddings = self.ques_channel(embeds)
        vi = image_embeddings
        u = ques_embeddings
        for att_layer in self.san:
            u = att_layer(vi, u)
        output = self.mlp(u)
        return output
