# src/models.py

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Model Definition(s)
# -----------------------------

class Simple1DCNN(nn.Module):
    """
    A simple 1D Convolutional Neural Network for time series classification.
    Consists of multiple convolutional blocks followed by fully connected layers.
    Uses BatchNorm, Max Pooling, Dropout, ReLU activations, and Adaptive Average Pooling.
    """
    def __init__(self, input_channels, num_classes):
        """
        Initializes the layers of the CNN.

        Args:
            input_channels (int): The number of input features (channels) in the time series window.
                                  Corresponds to the number of sensors/derived features.
            num_classes (int): The number of output classes (activities).
        """
        super(Simple1DCNN, self).__init__()
        # Validate inputs
        if input_channels <= 0:
             raise ValueError(f"input_channels must be positive, got {input_channels}")
        if num_classes <= 0:
             raise ValueError(f"num_classes must be positive, got {num_classes}")

        logging.info(f"Initializing Simple1DCNN model with input_channels={input_channels}, num_classes={num_classes}")

        # --- Convolutional Block 1 ---
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5, padding=2)
        # Padding='same' equivalent for kernel_size=5: padding=2
        self.bn1 = nn.BatchNorm1d(64) # Batch normalization for stabilizing training
        self.pool1 = nn.MaxPool1d(kernel_size=2) # Max pooling to reduce sequence length
        self.dropout1 = nn.Dropout(0.3) # Dropout for regularization

        # --- Convolutional Block 2 ---
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.3)

        # --- Convolutional Block 3 ---
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2) # Increased filter count
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.3)

        # --- Global Average Pooling ---
        # Reduces the sequence dimension to 1, making the model less sensitive to input length variations
        self.gap = nn.AdaptiveAvgPool1d(1)

        # --- Fully Connected (Dense) Layers ---
        self.fc1 = nn.Linear(256, 128) # Input size matches the out_channels of the last conv block after GAP
        self.dropout_fc1 = nn.Dropout(0.3) # Dropout before the next layer
        # Removed intermediate layers for simplicity (can be added back if needed)
        # self.fc2 = nn.Linear(128, 64)
        # self.dropout_fc2 = nn.Dropout(0.3)
        # self.fc3 = nn.Linear(64, 128)
        # self.dropout_fc3 = nn.Dropout(0.3)
        # self.fc4 = nn.Linear(128, 128)
        # self.dropout_fc4 = nn.Dropout(0.3)
        self.fc_out = nn.Linear(128, num_classes) # Final output layer producing class logits

        logging.info("Simple1DCNN model layers initialized.")

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes) containing raw logits.
        """
        # Input shape check (optional debug)
        # logging.debug(f"Model input shape: {x.shape}")

        # Apply Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        # logging.debug(f"After block 1 shape: {x.shape}")

        # Apply Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        # logging.debug(f"After block 2 shape: {x.shape}")

        # Apply Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        # logging.debug(f"After block 3 shape: {x.shape}")

        # Apply Global Average Pooling
        x = self.gap(x) # Shape becomes (batch_size, 256, 1)
        # logging.debug(f"After GAP shape: {x.shape}")

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1) # Shape becomes (batch_size, 256)
        # logging.debug(f"After flatten shape: {x.shape}")

        # Apply Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x) # Apply activation
        x = self.dropout_fc1(x)
        # logging.debug(f"After FC1 shape: {x.shape}")

        # Apply final output layer (no activation here, as CrossEntropyLoss expects logits)
        x = self.fc_out(x)
        # logging.debug(f"Final output shape: {x.shape}")

        return x
    
class FEN(nn.Module):
    """Feature Extraction Network (FEN) using 1D CNNs - Processes ONE channel at a time."""
    def __init__(self, orig_in_channels, out_channels1, out_channels2, out_channels3, out_channels4):
        super(FEN, self).__init__()
        kernel_size = 3
        padding = 1
        maxpool_kernel_size = 2
        dropout_rate = 0.2

        logging.info(f"Initializing FEN (will process {orig_in_channels} channels sequentially). CNN input channel = 1.")
        # <<< MODIFIED: Conv1d in_channels is always 1 >>>
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, out_channels1, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=2),
            nn.Dropout(dropout_rate)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(out_channels1, out_channels2, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=2),
            nn.Dropout(dropout_rate)
        )
        # <<< MODIFIED: Added missing blocks based on sequential loops >>>
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(out_channels2, out_channels3, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=2),
            nn.Dropout(dropout_rate)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv1d(out_channels3, out_channels4, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=2),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x_single_channel):
        # Expects input shape: (batch, 1, seq_len)
        if x_single_channel.shape[1] != 1:
             logging.warning(f"FEN received input with {x_single_channel.shape[1]} channels, expected 1. Check model forward pass.")
             # Attempt to process first channel only as fallback? Or raise error?
             # For now, let's proceed assuming the calling code handles it.
             # raise ValueError("FEN expects input with 1 channel for sequential processing.")

        x = self.conv_block1(x_single_channel)
        x = self.conv_block2(x)
        # <<< MODIFIED: Pass through all blocks >>>
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        # <<< END MODIFIED >>>
        # Output shape: (batch, out_channels4, seq_len_out)
        # Permute is handled in the wrapper model now after concatenation
        return x

class ResBLSTM(nn.Module):
    """Residual Bidirectional LSTM Layer."""
    # ... (Keep existing ResBLSTM code - no changes needed here) ...
    def __init__(self, input_size, hidden_size, num_layers):
        super(ResBLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.transform = nn.Linear(input_size, hidden_size * 2)
        logging.debug("ResBLSTM initialized.") # Changed to debug

    def forward(self, x):
        residual = self.transform(x)
        output, (hn, cn) = self.lstm(x)
        output = self.layer_norm(output)
        return output + residual

class AttentionLayer(nn.Module):
    """Attention mechanism layer."""
    def __init__(self, input_size): # Input is the feature size from LSTM
        super(AttentionLayer, self).__init__()
        # Attention mechanism
        self.attention_weights_layer = nn.Linear(input_size, 1)
        logging.info("AttentionLayer initialized.")

    def forward(self, x):
        # x shape: (batch, seq_len, input_size which is hidden_size * 2)
        # Calculate attention scores
        attention_scores = self.attention_weights_layer(x).squeeze(-1) # -> (batch, seq_len)

        # Apply softmax to get weights
        attention_weights = F.softmax(attention_scores, dim=1) # -> (batch, seq_len)

        # Calculate weighted sum of features
        # attention_weights.unsqueeze(1) -> (batch, 1, seq_len)
        # torch.bmm((batch, 1, seq_len), (batch, seq_len, input_size)) -> (batch, 1, input_size)
        weighted_feature_vector = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1) # -> (batch, input_size)

        return weighted_feature_vector, attention_weights # Return vector and weights (optional)

class FLN(nn.Module):
    """Feature Learning Network (FLN) using ResBLSTM and Attention."""
    # <<< MODIFIED: Input size depends on FEN output and number of original signals >>>
    def __init__(self, combined_fen_output_size, hidden_size, num_lstm_layers, num_classes):
        super(FLN, self).__init__()
        logging.info(f"Initializing FLN with input_size={combined_fen_output_size}, hidden={hidden_size}, classes={num_classes}")
        lstm_output_size = hidden_size * 2 # Bidirectional
        # Input size to ResBLSTM is the concatenated feature dimension
        self.res_bilstm = ResBLSTM(combined_fen_output_size, hidden_size, num_layers=num_lstm_layers)
        self.attention_layer = AttentionLayer(lstm_output_size)
        self.fc = nn.Linear(lstm_output_size, num_classes)
        logging.info("FLN initialized.")

    def forward(self, x_combined_fen):
        # x_combined_fen shape: (batch, seq_len_out, combined_fen_output_size)
        x = self.res_bilstm(x_combined_fen)    # -> (batch, seq_len_out, hidden_size * 2)
        attention_output, _ = self.attention_layer(x) # -> (batch, hidden_size * 2)
        classification_output = self.fc(attention_output) # -> (batch, num_classes)
        return classification_output

# --- Wrapper Model ---
class GeARFEN(nn.Module):
    """Combines FEN and FLN using sequential FEN processing."""

    def __init__(self, num_classes, num_original_features, # Takes num_original_features directly
                 fen_out_channels1, fen_out_channels2, fen_out_channels3, fen_out_channels4,
                 fln_hidden_size, fln_num_lstm_layers):
        super(GeARFEN, self).__init__()
        logging.info(f"Initializing GeARFEN (expects {num_original_features} features sequentially)...")
        self.num_signals = num_original_features # Store original feature count

        # FEN always takes in_channels=1 for this strategy
        self.fen = FEN(orig_in_channels=self.num_signals, # Pass original count for info log
                       out_channels1=fen_out_channels1,
                       out_channels2=fen_out_channels2,
                       out_channels3=fen_out_channels3,
                       out_channels4=fen_out_channels4)

        # FLN input size is FEN's last output channel count * number of original signals concatenated
        fln_input_size = fen_out_channels4 * self.num_signals
        self.fln = FLN(combined_fen_output_size=fln_input_size,
                       hidden_size=fln_hidden_size,
                       num_lstm_layers=fln_num_lstm_layers,
                       num_classes=num_classes)
        logging.info("GeARFEN initialized.")

    def forward(self, x):
        # x shape: (batch, num_original_features, seq_len) - IMPORTANT ASSUMPTION
        # This model's forward pass *requires* the input x to have channels == num_original_features
        batch_size, num_input_channels, seq_len = x.shape

        # Check if the input received matches the expected number of original features
        if num_input_channels != self.num_signals:
            logging.error(f"Input tensor channel dimension ({num_input_channels}) does not match the expected number of original features ({self.num_signals}) for sequential processing in GeARFEN.")
            raise ValueError(f"GeARFEN forward expects {self.num_signals} input channels, but received {num_input_channels}.")

        fen_outputs = []
        # Loop through each original signal/channel
        for i in range(self.num_signals):
            signal_input = x[:, i, :].unsqueeze(1) # (batch, 1, seq_len)
            fen_output = self.fen(signal_input) # (batch, fen_out_channels4, seq_len_out)
            fen_outputs.append(fen_output)

        # Concatenate FEN outputs along the channel dimension
        x_combined = torch.cat(fen_outputs, dim=1) # (batch, num_signals * fen_out_channels4, seq_len_out)
        # Permute for FLN/LSTM: -> (batch, seq_len_out, num_signals * fen_out_channels4)
        x_permuted = x_combined.permute(0, 2, 1)
        # Pass through FLN
        x_final = self.fln(x_permuted) # -> (batch, num_classes)
        return x_final

# --- Example Usage ---
if __name__ == '__main__':
    # ... (keep existing Simple1DCNN example usage) ...

    print("\n" + "="*50)
    print("--- CNNBiLSTMAttnModel Example ---")
    logging.basicConfig(level=logging.INFO)

    # Example instantiation using parameters (mimicking config)
    input_channels_ex = 50 # Example: number of features from data_prep summary
    num_classes_ex = 12   # Example: number of activities from data_prep summary
    window_size_ex = 100  # Example: sequence length

    model_params_ex = {
        'fen_out_channels1': 64, # Example values, adjust based on config/needs
        'fen_out_channels2': 128,
        'fen_out_channels3': 256,
        'fen_out_channels4': 128,
        'fln_hidden_size': 128, # Example
        'fln_num_lstm_layers': 2
    }

    new_model_instance = GeARFEN(
        input_channels=input_channels_ex,
        num_classes=num_classes_ex,
        **model_params_ex # Unpack parameters from dict
    )
    print("\n--- New Model Architecture ---")
    print(new_model_instance)

    # Optional: Use torchinfo for a detailed summary
    try:
        from torchinfo import summary
        example_input_shape = (32, input_channels_ex, window_size_ex) # Batch=32
        print("\n--- New Model Summary (torchinfo) ---")
        summary(new_model_instance, input_size=example_input_shape[1:]) # Pass (C, L)
    except ImportError:
        print("\nInstall 'torchinfo' for a detailed model summary.")
    except Exception as e:
         print(f"\nCould not generate torchinfo summary: {e}")

    # Example forward pass
    print("\n--- New Model Example Forward Pass ---")
    dummy_input = torch.randn(32, input_channels_ex, window_size_ex)
    print(f"Dummy input shape: {dummy_input.shape}")
    try:
        new_model_instance.eval()
        with torch.no_grad():
            output = new_model_instance(dummy_input)
        print(f"Output shape: {output.shape}")
        # Output shape should be (batch_size, num_classes), e.g., (32, 12)
        assert output.shape == (32, num_classes_ex)
        print("Forward pass successful.")
    except Exception as e:
        print(f"Error during example forward pass: {e}")