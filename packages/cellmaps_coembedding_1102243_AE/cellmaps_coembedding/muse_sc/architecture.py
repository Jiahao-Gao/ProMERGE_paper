# Classes used for coembedding
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# def cos_sim(A, B):
#         cosine = np.dot(A,B)/(norm(A)*norm(B))
#         return cosine

class ToTensor:
    """
    Convert numpy arrays to PyTorch tensors.
    """
    def __call__(self, sample):
        """
        Converts a sample from a numpy array to a floating point tensor.

        :param sample: A numpy array.
        :type sample: numpy.ndarray
        :return: Tensor converted from the numpy array.
        :rtype: torch.Tensor
        """
        return torch.from_numpy(sample).float()


class Protein_Dataset(Dataset):
    """
    A dataset class for storing protein data for training in PyTorch.

    :param data_train_x: Input features for training.
    :type data_train_x: numpy.ndarray
    :param data_train_y: Target outputs for training.
    :type data_train_y: numpy.ndarray
    """
    def __init__(self, data_train_x, data_train_y):
        self.data_train_x = data_train_x
        self.data_train_y = data_train_y

    def __len__(self):
        """
        Returns the size of the dataset.

        :return: Number of items in the dataset.
        :rtype: int
        """
        return len(self.data_train_x)

    def __getitem__(self, item):
        """
        Retrieves an item by its index.

        :param item: Index of the item.
        :type item: int
        :return: A tuple containing input features, target outputs, and the item index.
        :rtype: tuple
        """
        return self.data_train_x[item], self.data_train_y[item], item


def init_weights(m):
    """
    Initialize weights for linear layers using Xavier normal initialization and biases to zero.

    :param m: A PyTorch module.
    :type m: torch.nn.Module
    """
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight.data)
        #   nn.init.kaiming_normal(m.weight.data, nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0)


def init_weights_d(m):
    """
    Initialize weights for linear layers using normal distribution.

    :param m: A PyTorch module.
    :type m: torch.nn.Module
    """
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight.data)


class structured_embedding(nn.Module):
    """
    A PyTorch module for structured embedding of proteins using deep learning.

    :param x_input_size: Size of the input feature vector for x.
    :type x_input_size: int
    :param y_input_size: Size of the input feature vector for y.
    :type y_input_size: int
    :param latent_dim: Dimensionality of the latent space.
    :type latent_dim: int
    :param hidden_size: Size of the hidden layers.
    :type hidden_size: int
    :param dropout: Dropout rate for regularization.
    :type dropout: float
    :param l2_norm: Whether to apply L2 normalization on the embeddings.
    :type l2_norm: bool
    """
    def __init__(self, x_input_size, y_input_size, latent_dim, hidden_size, dropout, l2_norm):
        super().__init__()

        self.l2_norm = l2_norm

        self.encoder_x = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(x_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh())

        self.encoder_y = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(y_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh())

        self.encoder_z = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size + hidden_size, latent_dim),
            nn.BatchNorm1d(latent_dim))

        self.decoder_h_x = nn.Linear(latent_dim, latent_dim, bias=False)
        self.decoder_h_y = nn.Linear(latent_dim, latent_dim, bias=False)

        self.decoder_x = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, x_input_size))

        self.decoder_y = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, y_input_size))

        # initialize weights
        self.encoder_x.apply(init_weights)
        self.encoder_y.apply(init_weights)
        self.encoder_z.apply(init_weights)
        self.decoder_h_x.apply(init_weights_d)
        self.decoder_h_y.apply(init_weights_d)
        self.decoder_x.apply(init_weights)
        self.decoder_y.apply(init_weights)

    def forward(self, x, y):
        """
        Forward pass through the network.

        :param x: Input features for x.
        :type x: torch.Tensor
        :param y: Input features for y.
        :type y: torch.Tensor
        :return: Tuple containing latent embeddings, reconstructed x and y, and hidden representations of x and y.
        :rtype: tuple
        """
        h_x = self.encoder_x(x)
        h_y = self.encoder_y(y)

        h = torch.cat((h_x, h_y), 1)
        z = self.encoder_z(h)

        # unit sphere
        if self.l2_norm:
            z = nn.functional.normalize(z, p=2, dim=1)

        z_x = self.decoder_h_x(z)
        z_y = self.decoder_h_y(z)

        x_hat = self.decoder_x(z_x)
        y_hat = self.decoder_y(z_y)

        return z, x_hat, y_hat, h_x, h_y
