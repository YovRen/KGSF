import json
import math
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from torch_geometric.nn.conv.gcn_conv import GCNConv

NEAR_INF = 1e20
NEAR_INF_FP16 = 65504

class MyCLUB(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MyCLUB, self).__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, y_dim)
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, y_dim),
            nn.Tanh()
        )
        self.relu = nn.ReLU()

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples, z_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # Log-likelihood of positive sample pairs
        positive = - ((mu - y_samples)**2 + (mu - z_samples)**2) / (2. * logvar.exp())

        # Constructing pairs for negative log-likelihood
        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]
        z_samples_1 = z_samples.unsqueeze(0)  # shape [1,nsample,dim]

        # Log-likelihood of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2 + (z_samples_1 - prediction_1)**2).mean(dim=1) / (2. * logvar.exp())

        # CLUB estimation
        return (self.relu(positive.sum(dim=-1) - negative.sum(dim=-1))).mean()

    def loglikeli(self, x_samples, y_samples, z_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 / logvar.exp() - (mu - z_samples)**2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples, z_samples):
        return -self.loglikeli(x_samples, y_samples, z_samples)


def create_position_codes(n_pos, dim, out):
    position_enc = np.array([[pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)] for pos in range(n_pos)])
    sin_values = torch.FloatTensor(np.sin(position_enc)).type_as(out)
    cos_values = torch.FloatTensor(np.cos(position_enc)).type_as(out)
    with torch.no_grad():
        out[:, 0::2].copy_(sin_values)
        out[:, 1::2].copy_(cos_values)


def _normalize(tensor, norm_layer):
    size = tensor.size()
    return norm_layer(tensor.view(-1, size[-1])).view(size)


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super().__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, x):
        assert self.dim == x.shape[1]
        e = torch.matmul(torch.tanh(torch.matmul(x, self.a)), self.b).squeeze(dim=1)
        attention = F.softmax(e)
        return torch.matmul(attention, x)


class SelfAttentionLayer_batch(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super().__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, x, mask):
        assert self.dim == x.shape[2]
        mask = 1e-30 * mask.float()
        e = torch.matmul(torch.tanh(torch.matmul(x, self.a)), self.b)
        attention = F.softmax(e + mask.unsqueeze(-1), dim=1)
        return torch.matmul(torch.transpose(attention, 1, 2), x).squeeze(1), attention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout=0):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.attn_dropout = nn.Dropout(p=dropout)
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        self.out_lin = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(self, query, key=None, value=None, mask=None):
        batch_size, query_len, dim = query.size()
        assert dim == self.dim, f'Dimensions do not match: {dim} query vs {self.dim} configured'
        assert mask is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = tensor.transpose(1, 2).contiguous().view(batch_size * n_heads, seq_len, dim_per_head)
            return tensor

        if key is None and value is None:
            key = value = query
        elif value is None:
            value = key

        _, key_len, dim = key.size()
        q = prepare_head(self.q_lin(query))
        k = prepare_head(self.k_lin(key))
        v = prepare_head(self.v_lin(value))
        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        attn_mask = ((mask == 0).view(batch_size, 1, -1, key_len).repeat(1, n_heads, 1, 1).expand(batch_size, n_heads, query_len, key_len).view(batch_size * n_heads, query_len, key_len))
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill_(attn_mask, -NEAR_INF_FP16 if dot_prod.dtype is torch.float16 else -NEAR_INF)
        attn_weights = F.softmax(dot_prod, dim=-1).type_as(query)
        attn_weights = self.attn_dropout(attn_weights)
        attentioned = attn_weights.bmm(v)
        attentioned = (attentioned.type_as(query).view(batch_size, n_heads, query_len, dim_per_head).transpose(1, 2).contiguous().view(batch_size, query_len, dim))
        out = self.out_lin(attentioned)
        return out


class TransformerFFN(nn.Module):
    def __init__(self, dim, dim_hidden, relu_dropout=0):
        super().__init__()
        self.relu_dropout = nn.Dropout(p=relu_dropout)
        self.lin1 = nn.Linear(dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.relu_dropout(x)
        x = self.lin2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_heads, embedding_size, ffn_size, attention_dropout=0.0, relu_dropout=0.0, dropout=0.0):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.attention = MultiHeadAttention(n_heads, embedding_size, dropout=attention_dropout)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.ffn = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor, encoder_mask):
        tensor = tensor + self.dropout(self.attention(tensor, mask=encoder_mask))
        tensor = _normalize(tensor, self.norm1)
        tensor = tensor + self.dropout(self.ffn(tensor))
        tensor = _normalize(tensor, self.norm2)
        tensor *= encoder_mask.unsqueeze(-1).type_as(tensor)
        return tensor


class TransformerDecoder4KGLayer(nn.Module):
    def __init__(self, n_heads, embedding_size, ffn_size, attention_dropout=0.0, relu_dropout=0.0, dropout=0.0):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.dropout = nn.Dropout(p=dropout)
        self.self_attention = MultiHeadAttention(n_heads, embedding_size, dropout=attention_dropout)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.encoder_attention = MultiHeadAttention(n_heads, embedding_size, dropout=attention_dropout)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.encoder_db_attention = MultiHeadAttention(n_heads, embedding_size, dropout=attention_dropout)
        self.norm2_db = nn.LayerNorm(embedding_size)
        self.encoder_kg_attention = MultiHeadAttention(n_heads, embedding_size, dropout=attention_dropout)
        self.norm2_kg = nn.LayerNorm(embedding_size)
        self.ffn = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(self, x, encoder_output, encoder_mask, kg_encoder_output, kg_encoder_mask, db_encoder_output, db_encoder_mask):
        decoder_mask = self._create_selfattn_mask(x)
        residual = x
        x = self.self_attention(query=x, mask=decoder_mask)
        x = self.dropout(x)
        x = x + residual
        x = _normalize(x, self.norm1)
        # ori0: start
        residual = x
        x = self.encoder_db_attention(query=x, key=db_encoder_output, value=db_encoder_output, mask=db_encoder_mask)
        x = self.dropout(x)
        x = residual + x
        x = _normalize(x, self.norm2_db)
        residual = x
        x = self.encoder_kg_attention(query=x, key=kg_encoder_output, value=kg_encoder_output, mask=kg_encoder_mask)
        x = self.dropout(x)
        x = residual + x
        x = _normalize(x, self.norm2_kg)
        # ori0: end
        residual = x
        x = self.encoder_attention(query=x, key=encoder_output, value=encoder_output, mask=encoder_mask)
        x = self.dropout(x)
        x = residual + x
        x = _normalize(x, self.norm2)
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x
        x = _normalize(x, self.norm3)
        return x

    def _create_selfattn_mask(self, x):
        bsz = x.size(0)
        time = x.size(1)
        mask = torch.tril(x.new(time, time).fill_(1))
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask


class TransformerEncoder(nn.Module):
    def __init__(self, args, dictionary, embedding=None, padding_idx=None, reduction=True, n_positions=1024):
        super(TransformerEncoder, self).__init__()
        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.embedding_size = args.embedding_size
        self.ffn_size = args.ffn_size
        self.vocabulary_size = len(dictionary) + 4
        self.embedding = embedding
        self.p = args.dropout
        self.dropout = nn.Dropout(p=self.p)
        self.attention_dropout = args.attention_dropout
        self.relu_dropout = args.relu_dropout
        self.padding_idx = padding_idx
        self.reduction = reduction
        self.n_positions = n_positions
        self.dim = self.embedding_size
        self.out_dim = self.embedding_size
        assert self.embedding_size % self.n_heads == 0, 'Transformer embedding size must be a multiple of n_heads'
        self.embeddings = embedding
        self.position_embeddings = nn.Embedding(n_positions, self.embedding_size)
        create_position_codes(n_positions, self.embedding_size, out=self.position_embeddings.weight)
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerEncoderLayer(self.n_heads, self.embedding_size, self.ffn_size, attention_dropout=self.attention_dropout, relu_dropout=self.relu_dropout, dropout=self.p))  # mask1: self.layers.append(TransformerEncoderLayer(self.n_heads, self.embedding_size+128, self.ffn_size+128, attention_dropout=self.attention_dropout, relu_dropout=self.relu_dropout, dropout=self.dropout))

    def forward(self, input):
        mask = input != self.padding_idx  # kg2: mask是参数
        positions = (mask.cumsum(dim=1, dtype=torch.int64) - 1).clamp_(min=0)
        tensor = self.embeddings(input)  # kg3: tensor=input 没有self.embedding
        tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        # mask2: tensor=torch.cat([tensor,m_emb.unsqueeze(1).repeat(1,tensor.size()[1],1)],dim=-1)
        tensor = self.dropout(tensor)
        tensor *= mask.unsqueeze(-1).type_as(tensor)
        for i in range(self.n_layers):
            tensor = self.layers[i](tensor, mask)
        if self.reduction:
            divisor = mask.type_as(tensor).sum(dim=1).unsqueeze(-1).clamp(min=1e-7)
            output = tensor.sum(dim=1) / divisor
            return output
        else:
            output = tensor
            return output, mask


class TransformerDecoder4KG(nn.Module):
    def __init__(self, args, dictionary, embedding=None, padding_idx=None, n_positions=1024):
        super().__init__()

        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.embedding_size = args.embedding_size
        self.ffn_size = args.ffn_size
        self.vocabulary_size = len(dictionary) + 4
        self.embedding = embedding
        self.p = args.dropout
        self.dropout = nn.Dropout(p=self.p)
        self.attention_dropout = args.attention_dropout
        self.relu_dropout = args.relu_dropout
        self.padding_idx = padding_idx
        self.n_positions = n_positions
        self.dim = self.embedding_size
        self.out_dim = self.embedding_size
        self.dim = self.embedding_size
        self.out_dim = self.embedding_size
        assert self.embedding_size % self.n_heads == 0, 'Transformer embedding size must be a multiple of n_heads'
        self.embeddings = embedding
        self.position_embeddings = nn.Embedding(n_positions, self.embedding_size)
        create_position_codes(n_positions, self.embedding_size, out=self.position_embeddings.weight)
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerDecoder4KGLayer(self.n_heads, self.embedding_size, self.ffn_size, attention_dropout=self.attention_dropout, relu_dropout=self.relu_dropout, dropout=self.p))  # ori1: self.layers.append(TransformerDecoderLayer(self.n_heads, self.embedding_size, self.ffn_size, attention_dropout=self.attention_dropout, relu_dropout=self.relu_dropout, dropout=self.dropout))

    def forward(self, input, encoder_state, encoder_kg_state, encoder_db_state):
        encoder_output, encoder_mask = encoder_state
        kg_encoder_output, kg_encoder_mask = encoder_kg_state
        db_encoder_output, db_encoder_mask = encoder_db_state
        seq_len = input.size(1)
        positions = input.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)
        tensor = self.embeddings(input)
        tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.dropout(tensor)  # --dropout
        for layer in self.layers:
            tensor = layer(tensor, encoder_output, encoder_mask, kg_encoder_output, kg_encoder_mask, db_encoder_output, db_encoder_mask)  # ori2: tensor = layer(tensor, encoder_output, encoder_mask)
        return tensor


class CrossModel(nn.Module):
    def __init__(self, args, dictionary, pad_idx=0, start_idx=1, end_idx=2):
        super().__init__()
        self.batch_size = args.batch_size
        self.max_r_length = args.max_r_length
        self.end_idx = end_idx
        self.pad_idx = pad_idx
        self.device = args.device
        self.n_concept = args.n_concept
        self.n_entity = args.n_entity
        self.n_user = args.n_user
        self.n_relation = args.n_relation
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.dbpedia_subkg = args.dbpedia_subkg
        self.mask4key = torch.Tensor(np.load('data/mask4key.npy')).to(self.device)
        self.mask4movie = torch.Tensor(np.load('data/mask4movie.npy')).to(self.device)
        self.mask4 = self.mask4key + self.mask4movie
        self.n_positions = 1024
        self.hidden_dim = args.hidden_dim
        self.embedding_size = args.embedding_size
        # 生成部分db_attn，encoder_states_kg部分的参数
        self.concept_GCN = GCNConv(self.hidden_dim, self.hidden_dim)
        self.dbpedia_edge_sets = self._edge_list().to(self.device)
        self.db_edge_idx = self.dbpedia_edge_sets[:, :2].t()
        self.db_edge_type = self.dbpedia_edge_sets[:, 2]
        self.concept_edge_sets = self.concept_edge_list4GCN()
        self.dbpedia_RGCN = RGCNConv(self.n_entity + self.n_user, self.hidden_dim, self.n_relation, num_bases=8)
        self.concept_embeddings = nn.Embedding(args.n_concept + 1, self.hidden_dim)
        nn.init.normal_(self.concept_embeddings.weight, mean=0, std=self.hidden_dim ** -0.5)
        nn.init.constant_(self.concept_embeddings.weight[0], 0)
        self.con_graph_attn = SelfAttentionLayer_batch(self.hidden_dim, self.hidden_dim)
        self.user_graph_attn = SelfAttentionLayer_batch(self.hidden_dim, self.hidden_dim)
        self.db_graph_attn = SelfAttentionLayer(self.hidden_dim, self.hidden_dim)
        # mi_loss部分的参数
        self.club_mi = MyCLUB(self.hidden_dim, self.hidden_dim, self.hidden_dim)
        # rec_loss部分的参数
        self.user_fc = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.gate1_fc = nn.Linear(self.hidden_dim, 1)
        self.gate2_fc = nn.Linear(self.hidden_dim, 1)
        self.criterion = nn.CrossEntropyLoss(reduce=False)
        # sum_embeddings
        self.embeddings = nn.Embedding(len(dictionary) + 4, self.embedding_size, self.pad_idx)
        self.embeddings.weight.data.copy_(torch.from_numpy(np.load('data/word2vec_redial.npy')))
        # rec2_loss部分的参数
        self.encoder = TransformerEncoder(args, dictionary, self.embeddings, self.pad_idx, reduction=False, n_positions=self.n_positions)
        # gen_loss部分的参数
        self.con_graph_fc = nn.Linear(self.hidden_dim, self.embedding_size)
        self.db_graph_fc = nn.Linear(self.hidden_dim, self.embedding_size)
        self.con_graph_attn_fc = nn.Linear(self.hidden_dim, self.embedding_size)
        self.db_graph_attn_fc = nn.Linear(self.hidden_dim, self.embedding_size)
        self.decoder = TransformerDecoder4KG(args, dictionary, self.embeddings, self.pad_idx, n_positions=self.n_positions)
        self.decoder_graph_latent_fc = nn.Linear(self.embedding_size * 2 + self.embedding_size, self.embedding_size)
        self.decoder_graph_latent_fc_gen_fc = nn.Linear(self.embedding_size, len(dictionary) + 4)
        self.graph_rec_output = nn.Linear(self.hidden_dim, self.n_entity)

    def forward(self, context_vector, response_vector, concept_mask, db_mask, seed_sets, labels, concept_vector, dbpedia_vector, user_vector,  dbpedia_mentioned, user_mentioned, rec):
        dbpedia_nodes_features = self.dbpedia_RGCN(None, self.db_edge_idx, self.db_edge_type)
        db_nodes_features = dbpedia_nodes_features[:self.n_entity]
        user_nodes_features = dbpedia_nodes_features[self.n_entity:]
        con_nodes_features = self.concept_GCN(self.concept_embeddings.weight, self.concept_edge_sets)
        # 也是基础特征
        dbpedias_emb_list = []
        db_con_mask = []
        for i, seed_set in enumerate(seed_sets):
            if seed_set == []:
                dbpedias_emb_list.append(torch.zeros(self.hidden_dim).to(self.device))
                db_con_mask.append(torch.zeros([1]))
                continue
            db_graph_emb = db_nodes_features[seed_set]
            db_graph_attn_emb = self.db_graph_attn(db_graph_emb)
            dbpedias_emb_list.append(db_graph_attn_emb)
            db_con_mask.append(torch.ones([1]))
        db_graph_attn_emb = torch.stack(dbpedias_emb_list)
        con_graph_emb = con_nodes_features[concept_mask]
        con_graph_attn_emb, attention = self.con_graph_attn(con_graph_emb, (concept_mask == 0).to(self.device))
        user_graph_emb = user_nodes_features[user_mentioned]
        user_graph_attn_emb, attention = self.user_graph_attn(user_graph_emb, (user_mentioned == 0).to(self.device))
        # info_loss
        mi_loss = self.club_mi(user_graph_attn_emb, db_graph_attn_emb, con_graph_attn_emb)
        # 通过user_emb和db_nodes_features计算rec_scores，对比labels得到rec_loss
        uc_gate1 = F.sigmoid(self.gate1_fc(self.user_fc(torch.cat([con_graph_attn_emb, db_graph_attn_emb, user_graph_attn_emb], dim=-1))))
        uc_gate2 = F.sigmoid(self.gate2_fc(self.user_fc(torch.cat([con_graph_attn_emb, db_graph_attn_emb, user_graph_attn_emb], dim=-1))))
        user_emb = uc_gate1 * db_graph_attn_emb +uc_gate2 * con_graph_attn_emb+(1-uc_gate1-uc_gate2)*user_graph_attn_emb
        rec_scores = F.linear(user_emb, db_nodes_features, self.graph_rec_output.bias)
        rec_loss = torch.sum(self.criterion(rec_scores, labels.to(self.device)) * rec.float().to(self.device))
        # 计算gen_scores和preds--------可以把历史记录的movie_fc加上--------------------|##|Aab******#|Bc******#|Ac******#|Bc********#|-------------
        # generation---------------------------------------------------------------------------------------------------
        con_nodes_features4gen = con_nodes_features
        con_emb4gen = con_nodes_features4gen[concept_mask]
        con_mask4gen = concept_mask != 0
        kg_encoding = (self.con_graph_fc(con_emb4gen), con_mask4gen.to(self.device))
        db_emb4gen = db_nodes_features[dbpedia_mentioned]  # batch*50*dim
        db_mask4gen = dbpedia_mentioned != 0
        db_encoding = (self.db_graph_fc(db_emb4gen), db_mask4gen.to(self.device))
        encoder_states = self.encoder(context_vector)
        if response_vector != None:
            inputs = response_vector.narrow(1, 0, response_vector.size(1) - 1)
            inputs = torch.cat([self.START.detach().expand(self.batch_size, 1), inputs], 1)
            decoder_latent_emb = self.decoder(inputs, encoder_states, kg_encoding, db_encoding)
            kg_attention_latent = self.con_graph_attn_fc(con_graph_attn_emb)
            db_attention_latent = self.db_graph_attn_fc(db_graph_attn_emb)
            decoder_graph_latent_fc_emb = self.decoder_graph_latent_fc(torch.cat([kg_attention_latent.unsqueeze(1).repeat(1, response_vector.size(1), 1), db_attention_latent.unsqueeze(1).repeat(1, response_vector.size(1), 1), decoder_latent_emb], -1))
            decoder_graph_latent_fc_gen_fc_emb = self.decoder_graph_latent_fc_gen_fc(decoder_graph_latent_fc_emb) * self.mask4.unsqueeze(0).unsqueeze(0)
            decoder_latent_gen_scores = F.linear(decoder_latent_emb, self.embeddings.weight)
            sum_gen_scores = decoder_latent_gen_scores + decoder_graph_latent_fc_gen_fc_emb
            _, preds = sum_gen_scores.max(dim=2)
            scores = decoder_latent_gen_scores
            gen_loss = torch.mean(self.criterion(scores.view(-1, scores.size(-1)).to(self.device), response_vector.view(-1).to(self.device)))
        else:
            predict_vector = self.START.detach().expand(self.batch_size, 1)
            logits = []
            for i in range(self.max_r_length):
                decoder_latent_emb = self.decoder(predict_vector, encoder_states, kg_encoding, db_encoding)
                last_token_emb = decoder_latent_emb[:, -1:, :]
                con_graph_attn_fc = self.con_graph_attn_fc(db_graph_attn_emb)
                db_graph_attn_fc = self.db_graph_attn_fc(con_graph_attn_emb)
                decoder_graph_latent_fc_emb = self.decoder_graph_latent_fc(torch.cat([con_graph_attn_fc.unsqueeze(1), db_graph_attn_fc.unsqueeze(1), last_token_emb], -1))
                decoder_graph_latent_fc_gen_fc_emb = self.decoder_graph_latent_fc_gen_fc(decoder_graph_latent_fc_emb) * self.mask4.unsqueeze(0).unsqueeze(0)
                decoder_latent_gen_scores = F.linear(last_token_emb, self.embeddings.weight)
                sum_gen_scores = decoder_latent_gen_scores + decoder_graph_latent_fc_gen_fc_emb  # * (1 - gate)
                _, preds = sum_gen_scores.max(dim=-1)
                logits.append(sum_gen_scores)
                predict_vector = torch.cat([predict_vector, preds], dim=1)
                all_finished = ((predict_vector == self.end_idx).sum(dim=1) > 0).sum().item() == self.batch_size
                if all_finished:
                    break
            logits = torch.cat(logits, 1)
            scores = logits
            preds = predict_vector
            gen_loss = None
        return scores, preds, rec_scores, rec_loss, gen_loss, mi_loss

    def _edge_list(self):
        edge_list = []
        for dbpediaId in self.dbpedia_subkg:
            for (related, relation) in self.dbpedia_subkg[dbpediaId]:
                edge_list.append((int(dbpediaId), int(related), relation))
                edge_list.append((int(related), int(dbpediaId), relation))
        return torch.tensor(edge_list, dtype=torch.long)

    def concept_edge_list4GCN(self):
        node2index = json.load(open('data/key2index_3rd.json', encoding='utf-8'))
        f = open('data/conceptnet_edges2nd.txt', encoding='utf-8')
        edges = set()
        stopwords = set([word.strip() for word in open('data/stopwords.txt', encoding='utf-8')])
        for line in f:
            lines = line.strip().split('\t')
            entity0 = node2index[lines[1].split('/')[0]]
            entity1 = node2index[lines[2].split('/')[0]]
            if lines[1].split('/')[0] in stopwords or lines[2].split('/')[0] in stopwords:
                continue
            edges.add((entity0, entity1))
            edges.add((entity1, entity0))
        edge_set = [[co[0] for co in list(edges)], [co[1] for co in list(edges)]]
        return torch.LongTensor(edge_set).to(self.device)

    def freeze_kg(self, freezeKG):
        if freezeKG:
            params = [self.dbpedia_RGCN.parameters(), self.concept_GCN.parameters(), self.concept_embeddings.parameters(), self.con_graph_attn.parameters(), self.db_graph_attn.parameters(), self.db_graph_attn.parameters(), self.user_graph_attn.parameters(), self.user_db_con_fc.parameters(), self.graph_rec_output.parameters()]
            for param in params:
                for pa in param:
                    pa.requires_grad = False
            print(f"Freeze parameters in the model")
        else:
            params = [self.dbpedia_RGCN.parameters(), self.concept_GCN.parameters(), self.concept_embeddings.parameters(), self.con_graph_attn.parameters(), self.db_graph_attn.parameters(), self.db_graph_attn.parameters(), self.user_graph_attn.parameters(), self.user_db_con_fc.parameters(), self.graph_rec_output.parameters()]
            for param in params:
                for pa in param:
                    pa.requires_grad = True
            print(f"UnFreeze parameters in the model")

    def save_model(self, tag):
        if tag == "rec":
            torch.save(self.state_dict(), 'rec_net_parameter.pkl')
        else:
            torch.save(self.state_dict(), 'gen_net_parameter.pkl')

    def load_model(self, tag):
        if tag == "rec":
            self.load_state_dict(torch.load('rec_net_parameter.pkl'), strict=False)
        else:
            self.load_state_dict(torch.load('gen_net_parameter.pkl'), strict=False)
