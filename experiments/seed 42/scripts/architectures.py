import torch
import torch.nn as nn

class ResidualLin_Architecture(nn.Module):
    """
    A feed-forward network with skip (residual) connections.
    """
    def __init__(
        self,
        layers: list[int],
        activation: nn.Module | None = None
    ) -> None:
        """
        Initialize the network architecture with Xavier weight initialization.

        Parameters
        ----------
        layers : list of int
            Sizes of each layer, including input and output dimensions.
        activation : torch.nn.Module, optional
            Activation function to apply between layers (default: SiLU).
        
        Returns
        -------
        None
        """
        super().__init__()
        self.activation = activation if activation is not None else nn.SiLU()
        self.layers = nn.ModuleList(
            nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)
        )
        self.initialize_weights()
    
    def initialize_weights(self) -> None:
        """
        Initialize network weights using Xavier normal distribution and zero biases.

        Returns
        -------
        None
        """
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections every two layers.

        Parameters
        ----------
        x : torch.Tensor of shape (N, in_features)
            Input tensor containing concatenated [x, t, p1, p2] where (p1,p2) are the regime parameters (e.g., invariants).

        Returns
        -------
        torch.Tensor of shape (N, 2)
            Output tensor [h, q].
        """
        out = x
        for i, layer in enumerate(self.layers):
            out_new = self.activation(layer(out))
            if i % 2 == 1:
                out = out + out_new
            else:
                out = out_new
        return out

        

class Lin_Architecture(nn.Module):
    """
    A simple feed-forward network without skip connections.
    """
    def __init__(
        self,
        layers: list[int],
        activation: nn.Module | None = None
    ) -> None:
        """
        Initialize the network architecture with Xavier weight initialization.

        Parameters
        ----------
        layers : list of int
            Sizes of each layer, including input and output dimensions.
        activation : torch.nn.Module, optional
            Activation function to apply between hidden layers (default: SiLU).

        Returns
        -------
        None
        """
        super().__init__()
        self.activation = activation if activation is not None else nn.SiLU()
        self.layers = nn.ModuleList(
            nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)
        )
        self.initialize_weights()
    
    def initialize_weights(self) -> None:
        """
        Initialize network weights using Xavier normal distribution and zero biases.

        Returns
        -------
        None
        """
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all layers, applying activation except on final layer.

        Parameters
        ----------
        x : torch.Tensor of shape (N, in_features)
            Input tensor containing concatenated [x, t, p1, p2] where (p1,p2) are the regime parameters (e.g., invariants).

        Returns
        -------
        torch.Tensor of shape (N, 2)
            Output tensor [h, q].
        """
        out = x
        # apply activation on all but last layer
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        # final linear layer
        out = self.layers[-1](out)
        return out


