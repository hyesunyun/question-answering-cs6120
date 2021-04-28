"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


class EmbeddingWithCharacter(nn.Module):
    """Embedding layer used by BiDAF, with the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(EmbeddingWithCharacter, self).__init__()
        self.drop_prob = drop_prob
        
        # word embedding
        self.word_emb_size = word_vectors.size(1)
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        
        # character embedding
        self.char_emb_size = char_vectors.size(1)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors, freeze=False)
        
        # CNN layer
        n_filters = self.word_emb_size
        kernel_size = 5
        self.cnn = CNN(self.char_emb_size, n_filters, k=kernel_size)
        
        self.proj = nn.Linear(2*self.word_emb_size, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x_word, x_char):
        """
        Return the embedding for the words in a batch of sentences.
        Computed from the concatenation of a word-based lookup embedding and a character-based CNN embedding
      
        Args:
            'x_word' (torch.Tensor): Tensor of integers of shape (batch_size, seq_len) where
                each integer is an index into the word vocabulary
            'x_char' (torch.Tensor): Tensor of integers of shape (batch_size, seq_len, max_word_len) where
                each integer is an index into the character vocabulary
        Return:
            'emb' (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_size)containing the embeddings for each word of the sentences in the batch
        """
        
        # char embedding
        _, seq_len, max_word_len = x_char.size()
            # reshape to a batch of characters word-sequence
        x_char = x_char.view(-1, max_word_len)      # (b = batch_size*seq_len, max_word_len)
            # character-level embedding
        emb_char = self.char_embed(x_char)          # (b, max_word_len, char_emb_size)
            # transpose to match the CNN shape requirements
        emb_char = emb_char.transpose(1, 2)         # (b, n_channel_in = char_emb_size, max_word_len)
            # pass through cnn
        emb_char = self.cnn(emb_char)               # (b, n_channel_out = word_emb_size)
            # reshape to a batch of sentences of words embeddings
        emb_char = emb_char.view(-1, seq_len, self.word_emb_size)  # (batch_size, seq_len, word_emb_size)
    
        # word embedding
        emb_word = self.embed(x_word)               # (batch_size, seq_len, word_emb_size)
        
        # concatenate the char and word embeddings
        emb = torch.cat((emb_word, emb_char), 2)    # (batch_size, seq_len, 2*word_emb_size)
        
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)                        # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)                         # (batch_size, seq_len, hidden_size)

        return emb

class CNN(nn.Module):
    """Convolutional layer
    1st stage of computing a word embedding from its char embeddings
    
    Remark: process each word in the batch independently
    """
    
    def __init__(self, char_emb_size, f, k=5):
        """Init CNN
        
        Args:
            'char_emb_size' (int): character Embedding Size (nb of input channels)
            'f' (int): number of filters (nb of output channels)
            'k' (int, default=5): kernel (window) size
        """
        super(CNN, self).__init__()
        self.conv1D = nn.Conv1d(char_emb_size, f, k, bias=True)
     
    def forward(self, X_reshaped):
        """Compute the first stage of the word embedding
        
        Args:
            'X_reshaped' (Tensor, shape=(b, char_emb_size, max_word_length)): char-embedded words
                b = batch of words size
        
        Returns:
            'X_conv_out' (Tensor, shape=(b, f)): output of the convolutional layer
        """
        
        X_conv = self.conv1D(X_reshaped) # (b, f, max_word_length - k +1)
        
        # pooling layer to collapse the last dimension
        X_conv_out, _ = torch.max(F.relu(X_conv), dim=2) # (b, f)
                
        return X_conv_out


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s

# Version of Coattention that performed
# class Coattention(nn.Module):
#     """Coattention Layer.

#     Based on the paper:
#     "Dynamic Coattention Networks For Question Answering"
#     by Caiming Xiong, Victor Zhong, Richard Socher
#     (https://arxiv.org/abs/1611.01604).

#     Coattention computes attention in two directions and involves a second-level
#     attention computation (attending over representations that are themselves
#     attention outputs:
#     The first level attention is the C2Q attention which gets concatenated with
#     the second-level attention outputs. This is then fed through a bidirectional
#     LSTM.

#     The output has shape (batch_size, context_len, 2 * hidden_size).

#     Args:
#         hidden_size (int): Size of hidden activations.
#         drop_prob (float): Probability of zero-ing out activations.
#     """
#     def __init__(self, hidden_size, drop_prob=0.1):
#         super(Coattention, self).__init__()
#         self.drop_prob = drop_prob
#         self.hidden_size = hidden_size
#         # self.encoder = RNNEncoder(input_size=6*hidden_size,
#         #                              hidden_size=hidden_size,
#         #                              num_layers=1,
#         #                              drop_prob=drop_prob)
#         self.encoder = RNNEncoder(input_size=4*hidden_size,
#                                      hidden_size=hidden_size,
#                                      num_layers=1,
#                                      drop_prob=drop_prob)
#         self.q_linear = nn.Linear(hidden_size, hidden_size)
#         self.dropout = nn.Dropout(p=drop_prob)

#     def forward(self, c, q, c_mask, q_mask):

#         # second version based on description in project description pdf
#         qprime = torch.tanh(self.q_linear(q.view(-1, self.hidden_size))).view(q.size())
#         c_t = torch.transpose(c, 1, 2)
#         L = torch.bmm(qprime, c_t)
#         Alpha = F.softmax(L, dim=1)
#         a = torch.bmm(torch.transpose(Alpha, 1, 2), qprime)
#         Beta = F.softmax(L, dim=2)
#         b = torch.bmm(Beta, c)
#         s = torch.bmm(torch.transpose(Alpha, 1, 2), b)
#         bilstm_in = torch.cat((s, a), 2)
#         bilstm_in = self.dropout(bilstm_in)
#         U = self.encoder(bilstm_in, c_mask.sum(-1))
#         return U

class Coattention(nn.Module):
    """Coattention Layer.
    Based on the paper:
    "Dynamic Coattention Networks For Question Answering"
    by Caiming Xiong, Victor Zhong, Richard Socher
    (https://arxiv.org/abs/1611.01604).
    Coattention computes attention in two directions and involves a second-level
    attention computation (attending over representations that are themselves
    attention outputs:
    The first level attention is the C2Q attention which gets concatenated with
    the second-level attention outputs. This is then fed through a bidirectional
    LSTM.
    The output has shape (batch_size, context_len, 2 * hidden_size).
    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(Coattention, self).__init__()
        self.drop_prob = drop_prob
        self.hidden_size = hidden_size
        self.encoder = RNNEncoder(input_size=6*hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, c, q, c_mask, q_mask):

        # q encoding projection
        Qprime = torch.tanh(self.q_linear(q.view(-1, self.hidden_size))).view(q.size()) #B x n + 1 x l

        # co attention
        C_t = torch.transpose(c, 1, 2) #B x l x m + 1
        L = torch.bmm(Qprime, C_t) # L = B x n + 1 x m + 1

        A_Q_ = F.softmax(L, dim=1) # B x n + 1 x m + 1
        A_Q = torch.transpose(A_Q_, 1, 2) # B x m + 1 x n + 1
        C_Q = torch.bmm(C_t, A_Q) # (B x l x m + 1) x (B x m x n + 1) => B x l x n + 1

        A_C = F.softmax(L, dim=2)  # B x n + 1 x m + 1
        Q_t = torch.transpose(Qprime, 1, 2)  # B x l x n + 1
        C_C = torch.bmm(torch.cat((Q_t, C_Q), 1), A_C) # (B x l x n+1 ; B x l x n+1) x (B x n +1x m+1) => B x 2l x m + 1

        C_C_t = torch.transpose(C_C, 1, 2)  # B x m + 1 x 2l

        # BiLSTM
        bilstm_in = torch.cat((C_C_t, c), 2) # B x m + 1 x 3l
        bilstm_in = self.dropout(bilstm_in)
        #?? should it be d_lens + 1 and get U[:-1]
        U = self.encoder(bilstm_in, c_mask.sum(-1)) #B x m x 2l
        return U

class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2

class BiDAFCoattentionOutput(nn.Module):
    """Output layer used by BiDAF for question answering.
    This is used for BiDAF Coattention model since the ouput from attention
    layer is different for baseline attention layer.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFCoattentionOutput, self).__init__()
        self.att_linear_1 = nn.Linear(2 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(2 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
