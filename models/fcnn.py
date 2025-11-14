"""
Fully Connected Neural Network (FCNN) for Plant Health Classification

This module implements a traditional FCNN architecture for binary classification
of plant leaves (healthy vs. diseased).
"""

import torch
import torch.nn as nn


class FCNN(nn.Module):
    """
    Fully Connected Neural Network for image classification.
    
    Architecture:
        - Flatten input image to 1D vector
        - Multiple fully connected layers with ReLU activation
        - Dropout for regularization
        - Output layer with softmax for classification
    
    Args:
        input_size (int): Size of flattened input (default: 224*224*3 = 150528)
        hidden_sizes (list): List of hidden layer sizes (default: [2048, 1024, 512, 256])
        num_classes (int): Number of output classes (default: 2 for binary)
        dropout_rate (float): Dropout probability (default: 0.3)
    """
    
    def __init__(
        self, 
        input_size=224*224*3,
        hidden_sizes=[2048, 1024, 512, 256],
        num_classes=2,
        dropout_rate=0.3
    ):
        super(FCNN, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input images of shape [batch_size, channels, height, width]
        
        Returns:
            torch.Tensor: Class logits of shape [batch_size, num_classes]
        """
        # Flatten the input
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # [batch_size, input_size]
        
        # Pass through network
        output = self.network(x)  # [batch_size, num_classes]
        
        return output
    
    def get_num_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_fcnn_model(num_classes=2, dropout_rate=0.3):
    """
    Factory function to create FCNN model.
    
    Args:
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout probability
    
    Returns:
        FCNN: Initialized FCNN model
    """
    model = FCNN(
        input_size=224*224*3,
        hidden_sizes=[2048, 1024, 512, 256],
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    return model


if __name__ == "__main__":
    # Test the model
    model = create_fcnn_model(num_classes=2)
    
    # Print model architecture
    print("FCNN Model Architecture:")
    print("=" * 60)
    print(model)
    print("=" * 60)
    
    # Print parameter count
    num_params = model.get_num_parameters()
    print(f"\nTotal Parameters: {num_params:,}")
    print(f"Model Size: ~{num_params * 4 / (1024**3):.2f} GB (float32)")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\nInput Shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output Shape: {output.shape}")
    print(f"Output (logits):\n{output}")
    
    # Apply softmax to get probabilities
    probabilities = torch.softmax(output, dim=1)
    print(f"\nProbabilities:\n{probabilities}")
    
    # Get predictions
    predictions = torch.argmax(probabilities, dim=1)
    print(f"\nPredictions: {predictions}")
