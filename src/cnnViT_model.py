"""
This module defines the core architectural components for training and evaluating 
vision transformer (ViT) models tailored to image-to-ISD (Illuminant Spectral Distribution) 
regression tasks. It includes:

- A custom VisionTransformer class with flexible support for either:
    - Patch-based embeddings via Conv2d, or
    - CNN-based feature extraction (e.g., ResNet34 backbone)

- Modular positional encoding, patch embedding, and transformer encoder block definitions

- Support for both mean-guided (global ISD vector) and map-guided (pixel-wise ISD map) outputs

- A simple decoder for image reconstruction from transformer outputs

Class Overview
==============

    *   RearrangeLayer
        --------------
        A wrapper around `einops.rearrange` for use within nn.Sequential. 
        Allows tensor reshaping with a specified pattern.

    *   PositionalEncoding
        ------------------
        Implements sinusoidal positional encoding for sequence embeddings,
        based on the original Transformer paper.

    *   PatchEmbedding
        --------------
        Uses a Conv2D layer with stride equal to patch size to embed image patches,
        followed by positional encoding. Used when not using CNN feature extractors.

    *   CNNFeatureEmbedder
        ------------------
        A ResNet34-based CNN feature extractor that can be used in two modes:
        (1) as a patch embedding module for ViT or (2) as a standalone regression network.
        Supports positional encoding, pretrained weight loading, and freezing of CNN.

    *   MeanGuidanceOutputMLP
        ---------------------
        A flexible multi-layer perceptron (MLP) for regressing from a vector input to a 3D output.
        Used when `mean_guidance=True` for ViT output.

    *   SimpleDecoder
        -------------
        A transpose convolutional decoder that upsamples a sequence of patch embeddings 
        back to a full-resolution image. Used for ViT pixel-wise outputs.

    *   TransformerEncoderBlock
        -----------------------
        Implements a standard Transformer encoder block, including:
        LayerNorm, multi-head attention, MLP feedforward layers, and residual connections.

    *   VisionTransformer
        -----------------
        Main ViT model combining patch embedding (CNN-based or Conv2D),
        transformer encoder blocks, and output heads for regression or reconstruction.
        Supports mean-guidance (vector) or pixel-wise (image) output modes.


Usage:
------
This module is intended to be imported and used inside a PyTorch training pipeline. 
However, a testable `main()` function is provided for architecture sanity checks and debugging.

Example:
    >>> model = VisionTransformer(img_size=224, embed_dim=256, mean_guidance=True)
    >>> x = torch.randn(4, 3, 224, 224)
    >>> y = model(x)

"""


# Packages
import torch
import math
import logging

from src.map_ViT_models import ViTModel
from torch import nn, Tensor
from einops import rearrange, repeat
from torchvision.models import resnet34, resnet50, ResNet50_Weights


##############################################################################################################
## Utilities
##############################################################################################################
class RearrangeLayer(nn.Module):
    def __init__(self, pattern, **dims):
        """
        Custom layer for einops.rearrange.
        
        Args:
            pattern (str): Rearrangement pattern.
        """
        super().__init__()
        self.pattern = pattern
        self.dims = dims
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x):
        self.logger.debug(f"Rearranging tensor with pattern {self.pattern} and dimensions {self.dims}")
        return rearrange(x, self.pattern, **self.dims)
    
##############################################################################################################

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, max_len: int = 1000):
        """
        Sinusoidal Positional Encoding Module.

        Args:
            emb_size (int): The size of the embedding dimension.
            max_len (int): The maximum length of the sequence.
        """
        super(PositionalEncoding, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing PositionalEncoding with emb_size={emb_size}, max_len={max_len}")
              
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe = torch.zeros(max_len, emb_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, emb_size).

        Returns:
            Tensor: Positional encoded tensor of the same shape as input.
        """
        seq_len = x.size(1)
        self.logger.debug(f"Adding positional encoding to tensor of shape {x.shape}")
        
        # Ensure positional encoding is on the same device as x
        self.positional_encoding = self.positional_encoding.to(x.device)
        return x + self.positional_encoding[:, :seq_len, :]
    
##############################################################################################################

class PatchEmbedding(nn.Module):
    """
    Uses a COnv2 layer with stride=patch size to create the patch embeddings
    
    Args:
        in_channels (int): Number of channels in the input image.
        patch_size (int): The size of the patch to extract.
        embed_size (int): The size of the embedding dimension.
        img_size (int): SIze of the input iamge.
    """
    def __init__(self, in_channels: int, patch_size: int, embed_size: int, img_size: int):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing PatchEmbedding with in_channels={in_channels}, patch_size={patch_size}, embed_size={embed_size}, img_size={img_size}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"{self.__class__.__name__} sent to '{self.device}'.")  

       
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=embed_size, 
                      kernel_size=patch_size, 
                      stride=patch_size),
            RearrangeLayer('b e h w -> b (h w) e'),
        ).to(self.device)
        self.pos_embed = PositionalEncoding(embed_size, (img_size // patch_size)**2).to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Creates patch embedding of input iamge

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channel, image_size, image_size).

        Returns:
            Tensor: Conv2d with positional embedding output of shape ().
        """
        self.logger.debug(f"Input tensor shape: {x.shape}")
        x = x.to(self.device)
        x = self.embed(x)
        self.logger.debug(f"Conv2d output shape: {x.shape}")
        x = self.pos_embed(x)
        self.logger.debug(f"Output with positional encoding shape: {x.shape}")
        return x

##############################################################################################################

class CNNFeatureEmbedder(nn.Module):
    """
    CNN-based feature extractor for either ViT embedding or standalone regression.

    Args:
        embed (bool): If True, use model for ViT embedding with spatial feature maps.
                      If False, use model for standalone regression with global pooling.
        model_state_path (str): Optional path to pretrained model checkpoint.
        embed_dim (int): Output embedding dimension (default: 768).
        position_embedding (bool): If True, adds learnable positional embeddings.
    """
    def __init__(self, embed=True, model_state_path=None, input_size=512, embed_dim=768, position_embedding=True, freeze_backbone=False, dropout_rate=0.0):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.embed = embed
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.position_embedding = position_embedding
        self.model_state_path = model_state_path
        self.freeze_backbone = freeze_backbone
        self.dropout = nn.Dropout(p=dropout_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"Initializing CNNFeatureEmbedder | mode={'ViT' if embed else 'Regression'} | input_size={input_size} |embed_dim={embed_dim} | dropout_rate={dropout_rate} | device={self.device}")
        self._init_backbone()
        self._init_projection_layer()

        if self.embed and self.position_embedding:
            self.pos_encoder = PositionalEncoding(self.embed_dim).to(self.device)
        self._log_parameter_count()

    def _init_backbone(self):
        cnn = resnet34(weights=None if self.model_state_path else 'DEFAULT')
        # Load pretrained weights
        if self.model_state_path:
            self.logger.info(f"Loading model weights from: {self.model_state_path}")
            state_dict = torch.load(self.model_state_path, map_location='cpu')['model_state_dict']
            backbone_dict = {
                k.replace("feature_extractor.", ""): v
                for k, v in state_dict.items()
                if k.startswith("feature_extractor.")
            }
            self.feature_extractor = nn.Sequential(*list(cnn.children())[:-2])
            self.feature_extractor.load_state_dict(backbone_dict, strict=False)
        # Load default ImageNet trained weights
        else:
            self.logger.info(f"Loading default ImageNet model weights.")
            self.feature_extractor = nn.Sequential(*list(cnn.children())[:-2])
        for name, p in self.feature_extractor.named_parameters():
            p.requires_grad = not self.freeze_backbone
            self.logger.debug(f"Param: {name} | requires_grad={p.requires_grad}")
        # Init cnn backbone
        self.feature_extractor = self.feature_extractor.to(self.device)
        # If pretraining backbone pool and flatten layers prior to final fc layer
        if not self.embed: 
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
        self.logger.debug(f"CNN Backbone Architecture: \n{self.feature_extractor}")

    def _init_projection_layer(self):
        if self.embed:
            self.patch_embed = nn.Conv2d(512, self.embed_dim, kernel_size=1).to(self.device)
            self.logger.info(f"Initialized patch embedding layer with 1x1 convolution for ViT-compatible spatial embeddings:")
        else:
            self.patch_embed = nn.Linear(512, self.embed_dim).to(self.device)
            self.logger.info(f"Intialized patch embedding layer with linear projection for regression: \nInput Dims:")
        self.logger.info(f"Input Dims: {self.input_size} | Output Dims: {self.embed_dim}")
        self.logger.debug(f"Projection Layer Architecture: \n{self.patch_embed}")

    def _log_parameter_count(self):
        """
        Logs total and trainable parameters in the model, summarized by top-level modules.
        """
        self.logger.info(f"{self.__class__.__name__} Parameter Summary (Top-Level Modules):")
        self.logger.info("-" * 80)
        total_params = 0
        trainable_params = 0

        for name, module in self.named_children():  # Only top-level children
            mod_total = sum(p.numel() for p in module.parameters())
            mod_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += mod_total
            trainable_params += mod_trainable
            self.logger.info(f"{name:<25} | Total: {mod_total:<20} | Trainable: {mod_trainable:,}")

        self.logger.info("-" * 80)
        self.logger.info(f"Total Parameters:     {total_params:,}")
        self.logger.info(f"Trainable Parameters: {trainable_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CNN-based embedding module.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W)

        Returns:
            Tensor: Either (B, H*W, embed_dim) for ViT or (B, 3) for regression.
        """
        self.logger.debug(f"[Input] x shape: {x.shape}")
        x = x.to(self.device)
        features = self.feature_extractor(x)  # (B, 512, H, W)
        self.logger.debug(f"[Feature Extractor] Output shape: {features.shape}")

        if self.embed: # Patch embedding using Conv2D
            x = self.patch_embed(features)  # (B, embed_dim, H, W)
            x = self.dropout(x)
            self.logger.debug(f"[Patch Embed] Shape after Conv2D: {x.shape}")
            B, C, H, W = x.shape # Flatten spatial dimensions
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
            self.logger.debug(f"[Flatten + Transpose] Shape: {x.shape}")
            if self.position_embedding:
                x = self.pos_encoder(x)
                self.logger.debug(f"[Position Embedding] Shape: {x.shape}")
            return x
        else: # Global pooling and final projection
            pooled = self.global_pool(features)  # (B, 512, 1, 1)
            self.logger.debug(f"[Global Pool] Shape: {pooled.shape}")
            flat = self.flatten(pooled)  # (B, 512)
            self.logger.debug(f"[Flatten] Shape: {flat.shape}")
            flat = self.dropout(flat)
            out = self.patch_embed(flat)  # (B, 3)
            self.logger.debug(f"[Output] Final prediction shape: {out.shape}")
            return out
        

##############################################################################################################


class MeanGuidanceOutputMLP(nn.Module):
    def __init__(self, input_dim, output_dim=3, dropout_rate=0.5, decay_factor=4):
        """
        Fully connected MLP with exponential reduction.

        Args:
            input_dim (int): Input feature dimension.
            output_dim (int, optional): Output feature dimension (default: 3).
            dropout_rate (float, optional): Dropout probability (default: 0.5).
            decay_factor (int, optional): Exponential reduction factor (default: 4).
        """
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__} with input_dim={input_dim}, output_dim={output_dim}, dropout_rate={dropout_rate}, decay_factor={decay_factor}")

        # Compute exponential reduction for layer sizes
        hidden_dims = [input_dim]
        current_dim = input_dim

        while current_dim > output_dim:
            next_dim = max(current_dim // decay_factor, output_dim)
            hidden_dims.append(next_dim)
            current_dim = next_dim

        # Build MLP layers
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.LayerNorm(hidden_dims[i + 1]))  # More stable than BatchNorm for large inputs

        # Final output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.fc_layers = nn.Sequential(*layers)
        self.logger.debug(f"MLP Layer Structure:\n{self.fc_layers}")

    def forward(self, x):
        self.logger.debug(f"Input shape: {x.shape}")
        output = self.fc_layers(x)
        self.logger.debug(f"Output shape: {output.shape}")
        return output

##############################################################################################################
class SimpleDecoder(nn.Module):
    def __init__(self, in_channels=768, dropout=0.3):
        """
        Upsample ResNet feature map back to input size using transpose convolutions.

        Output Size = (Input Size - 1) x (Stride - 2) x (Padding + Kernel Size)

        Args:
            in_channels (int): Number of input channels from ResNet feature map.
            target_size (tuple): Target spatial size (height, width) to upsample to.
        """
        super().__init__()

        # Set up logger for the class
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing SimpleDecoder.")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Transpose convolution layers
        self.upsample = nn.Sequential(
            RearrangeLayer('b (h w) c -> b c h w', h=7, w=7),  # Reshape (batch, size, embed_dim) -> (batch, channel, height, width)

            nn.ConvTranspose2d(in_channels, 512, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),          # 14 -> 28
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),          # 28 -> 56
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),           # 56 -> 112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),             # 112 -> 224
        ).to(self.device)
        self.logger.info("SimpleDecoder initialized.")

    def forward(self, x):
        """
        Forward pass to upsample feature map to input size.
        
        Args:
            x (Tensor): Input feature map from ResNet of shape (batch_size, in_channels, height, width).
        
        Returns:
            Tensor: Upsampled feature map of shape (batch_size, 3, target_size[0], target_size[1]).
        """
        self.logger.debug(f"Forward pass started with input shape: {x.shape}")
        x = x.to(self.device)
        output = self.upsample(x)
        self.logger.debug(f"Forward pass completed. Output shape: {output.shape}")
        return output

##############################################################################################################

class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block (Vanilla ViT)
    Based on: https://arxiv.org/pdf/2010.11929.pdf
    """
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.5):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing TransformerEncoderBlock with embed_dim={embed_dim}, num_heads={num_heads}, dropout={dropout}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Self-attention layer
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        # Feedforward MLP
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),  # Dropout after activation
            nn.Linear(4 * embed_dim, embed_dim)
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        self._init_weights()
        self.to(self.device)

    def _init_weights(self):
        for module in self.feed_forward:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass of Transformer Encoder Block.

        Takes tensor input an runs through the following:
            * layer norm
            * Multihead Attention
            * Residula Connection
            * layer norm
            * multilayer perceptron
            * residual connection
        
        Args:
            x (Tensor): Input of shape (batch_size, num_patches, embed_dim)

        Returns:
            x (Tensor): Output of same shape as input.
        """

        self.logger.debug(f"Input shape: {x.shape}")

        # LayerNorm -> Self-Attention -> Residual Connection
        residual = x
        x = self.layer_norm1(x)  
        x, _ = self.mha(x, x, x)  # Multihead Self-Attention
        x = self.dropout(x) + residual  # Skip Connection
        
        # LayerNorm -> MLP -> Residual Connection
        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x) + residual  # Skip Connection

        self.logger.debug(f"Output shape: {x.shape}")
        return x

##############################################################################################################

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model with flexible embedding and output options.

    This model implements a Vision Transformer with the following features:
    - Patch embedding via either a CNN (ResNet34) or simple Conv2D-based patching.
    - Optional classifier token ([CLS]) for sequence-level regression tasks.
    - Configurable number of Transformer encoder blocks with optional external residual (skip) connections.
    - Support for both mean-guidance output (global ISD vector) and pixel-wise output (ISD map).
    - Optional freezing of CNN backbone and output layer for fine-tuning workflows.

    Args:
        num_layers (int): Number of Transformer encoder layers.
        img_size (int): Size of the input image (assumed square: img_size x img_size).
        embed_dim (int): Dimensionality of the embedding vectors.
        patch_size (int): Size of each image patch (used only if cnn_embedding=False).
        num_head (int): Number of attention heads in each Transformer block.
        trans_dropout (float): Dropout rate in Transformer blocks.
        output_dropout (float): Dropout rate applied in output projection layers.
        cnn_embedding (bool): If True, uses a ResNet34 CNN backbone for patch embedding; 
                              otherwise uses Conv2D-based patch projection.
        cnn_model_state_path (str or None): Optional path to pretrained CNN model checkpoint.
        mean_guidance (bool): If True, outputs a [B, 3] regression vector from the [CLS] token;
                              if False, produces a full [B, 3, H, W] output (ISD map).
        device (str or None): Computation device ('cuda' or 'cpu'). Defaults to GPU if available.
        freeze_cnn (bool): If True, freezes the CNN backbone during training.
        freeze_output_layer (bool): If True, freezes the output layer (e.g. during feature extraction).
        use_skip_con (bool): If True, applies explicit residual connections across Transformer blocks.
    """
    def __init__(self,
        num_layers=2,
        img_size=512,
        embed_dim=768,
        patch_size=16,
        num_head=8,
        trans_dropout=0.1,
        output_dropout=0.3,
        cnn_embedding=True,
        cnn_model_state_path=None,
        mean_guidance=True,
        device=None,
        freeze_cnn=False,
        freeze_output_layer=False,
        use_skip_con=False,
        use_cls_token=False):
        super().__init__()

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.mean_guidance = mean_guidance
        self.use_skip_con = use_skip_con

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__} with the following parameters:")
        self.logger.info(f"  - num_layers={num_layers}")
        self.logger.info(f"  - img_size={img_size}")
        self.logger.info(f"  - embed_dim={embed_dim}")
        self.logger.info(f"  - patch_size={patch_size}")
        self.logger.info(f"  - num_heads={num_head}")
        self.logger.info(f"  - trans_dropout={trans_dropout}")
        self.logger.info(f"  - output_dropout={output_dropout}")
        self.logger.info(f"  - cnn_embedding={cnn_embedding}")
        self.logger.info(f"  - cnn_model_state_path={cnn_model_state_path}")
        self.logger.info(f"  - freeze_cnn={freeze_cnn}")
        self.logger.info(f"  - mean_guidance={mean_guidance}")
        self.logger.info(f"  - freeze_output_layer={freeze_output_layer}")
        self.logger.info(f"  - use_skip_con={use_skip_con}")
        self.logger.info(f"  - use_cls_token={use_cls_token}")
        self.logger.info(f"{self.__class__.__name__} will run on device: {self.device}")

        # Classifier Token
        self.use_cls_token = use_cls_token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # Learnable [CLS] token
        if self.use_cls_token:
            self.logger.info(f"Initialized cls_token as: {self.cls_token.__class__.__name__} | Shape: {self.cls_token.shape}")

        # Patch Embedding
        if cnn_embedding:
            self.patch_emb = CNNFeatureEmbedder(
                embed=True,
                input_size=img_size,
                model_state_path=cnn_model_state_path,
                embed_dim=embed_dim,
                position_embedding=True,
                freeze_backbone=False
            )
        else:
            self.patch_emb = PatchEmbedding(
                in_channels=3,
                patch_size=patch_size,
                img_size=img_size,
                embed_size=embed_dim
            )
        if freeze_cnn:
            for param in self.patch_emb.parameters():
                param.requires_grad = False
        self.logger.info("PatchEmbedding layer frozen (requires_grad=False)")
        self.patch_emb.to(self.device)
        self.logger.info(f"Initialized Patch Embedding Layer: {self.patch_emb.__class__.__name__}")
        
        # Transformer Encoder
        self.logger.info(f"Initializing trans_encoder:")
        # self.trans_encoder = nn.Sequential(*[
        #     TransformerEncoderBlock(embed_dim, num_head, trans_dropout) for _ in range(num_layers)
        #     ]).to(self.device)
        self.trans_encoder = nn.ModuleList([
                TransformerEncoderBlock(embed_dim, num_head, trans_dropout)
                for _ in range(num_layers)
            ]).to(self.device)

        # Define Output Layer
        self.logger.info(f"Intializing Output Layer with mean_guidance={self.mean_guidance}, cnn_embedding={cnn_embedding}:")
        if mean_guidance:
            self.output_layer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),  # Hidden layer (half embedding size)
                nn.ReLU(),
                nn.Dropout(output_dropout),
                nn.Linear(embed_dim // 2, 3)  # Output layer (3D vector)
            ).to(self.device)
            self.logger.info(f"Using MLP for mean-guidance output: \n{self.output_layer}")
        else:
            if cnn_embedding:
                self.output_layer = SimpleDecoder(in_channels=embed_dim, dropout=output_dropout).to(self.device)
                self.logger.info(f"Using SimpleDecoder for pixel-wise output with input channels={embed_dim}")
            else:
                # Compute dynamic dimensions for rearranging patches
                h, w = img_size // patch_size, img_size // patch_size  # Grid size
                ph, pw = patch_size, patch_size  # Patch size
                self.output_layer = nn.Sequential(
                    RearrangeLayer('b (h w) (patch_c ph pw) -> b patch_c (h ph) (w pw)',
                                h=h, w=w, patch_c=3, ph=ph, pw=pw)
                ).to(self.device)
            self.logger.info(f"Using RearrangeLayer for reconstruction: patches ({h}x{w}) with patch size {ph}x{pw}")
        for p in self.output_layer.parameters():
            p.requires_grad = not freeze_output_layer
        self._log_parameter_count()

    def _log_parameter_count(self):
        """
        Logs total and trainable parameters in the model, summarized by top-level modules.
        """
        self.logger.info(f"{self.__class__.__name__} Parameter Summary (Top-Level Modules):")
        self.logger.info("-" * 80)

        total_params = 0
        trainable_params = 0

        for name, module in self.named_children():  # Only top-level children
            mod_total = sum(p.numel() for p in module.parameters())
            mod_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += mod_total
            trainable_params += mod_trainable
            self.logger.info(f"{name:<25} | Total: {mod_total:<20} | Trainable: {mod_trainable:,}")

        self.logger.info("-" * 80)
        self.logger.info(f"Total Parameters:     {total_params:,}")
        self.logger.info(f"Trainable Parameters: {trainable_params:,}")

    def forward(self, x):
        """
        Forward pass for the Vision Transformer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: 
                If mean_guidance=True: Output tensor of shape [B, 3]
                If mean_guidance=False: Reconstructed output image tensor of shape (batch_size, 3, img_size, img_size).
        """
        B = x.shape[0] # Batch size
        x = x.to(self.device)  # Move input tensor to the same device as the model
        self.logger.debug(f"Input shape: {x.shape}")

        x = self.patch_emb(x)
        self.logger.debug(f"Patch embeddings shape: {x.shape}")

        # Add classifier token
        cls_tokens = self.cls_token.expand(B, -1, -1).to(self.device)  # Expand to match batch size
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: [B, num_patches + 1, embed_d
        self.logger.debug(f"Patch embedding siwth classifier token shape: {x.shape}")

        # Transformer
        # x = self.trans_encoder(x)
        for block in self.trans_encoder:
            if self.use_skip_con:
                residual = x
                x = block(x)
                x = x + residual
            else:
                x = block(x)
        self.logger.debug(f"Transformer encoder output shape: {x.shape}")

        if self.mean_guidance:
            if self.use_cls_token:
                cls_output = x[:, 0, :]  # Shape: [B, embed_dim]
                self.logger.debug(f"Classifier token shape: {cls_output.shape}")
                output_vector = self.output_layer(cls_output)
            else:
                pooled_output = x[:, 1:, :].mean(dim=1)  # exclude CLS
                self.logger.debug(f"Pooled output shape: {pooled_output.shape}")
                output_vector = self.output_layer(pooled_output)
            self.logger.debug(f"{self.__class__.__name__} output shape: {output_vector.shape}")
            return output_vector

        x = x[:, 1:, :] # remove classifier token
        output_img = self.output_layer(x)
        self.logger.debug(f"Output layer output shape: {output_img.shape}")
        return output_img


##############################################################################################################
    
def main():

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    model = CNNFeatureEmbedder()
    logger.info("\n" + "==" * 80 + "\n")

    #Params
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_layers = 4
    embed_dim = 256
    num_head = 4
    patch_size=16
    img_size=224
    trans_dropout=0.1
    output_dropout=0.3
    cnn_embedding = True
    mean_guidance = True
    freeze_cnn = True
    freeze_output_layer = True
    use_skip_con=True
    cnn_model_state_path = "/home/massone.m/spectral_ratio/training/results/CNN_x100_log_04/model_states/best_model.pth"

    model = VisionTransformer(
                            num_layers=num_layers,
                            img_size=img_size,
                            embed_dim=embed_dim,
                            patch_size=patch_size,
                            num_head=num_head,
                            cnn_embedding=cnn_embedding,
                            mean_guidance=mean_guidance,
                            trans_dropout=trans_dropout,
                            output_dropout=output_dropout,
                            freeze_cnn = freeze_cnn,
                            freeze_output_layer = freeze_output_layer,
                            use_skip_con=use_skip_con,
                            cnn_model_state_path=cnn_model_state_path).to(device)

    # model = VisionTransformerPreTrained()
    # Test input
    batch_size = 4
    test_input = torch.rand(batch_size, 3, img_size, img_size).to(device)

    # Forward pass
    try:
        output = model(test_input)
        print("\nModel Output shape:", output.shape)
    except Exception as e:
        print(f"An error occurred during the forward pass: {e}")

if __name__ == "__main__":
    main()