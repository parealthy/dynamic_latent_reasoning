"""
Reasoning Network Hø(hidden states, soft prompt embeddings) -> reasoning embeddings
This module implements the Reasoning Network for the Latent Reasoning Model.

Inputs: hidden states, soft prompt embeddings
Outputs: logits for reasoning tasks
"""

import torch
import torch.nn as nn 
from transformers import AutoModel 

class ReasoningNet(torch.nn.Module):
    def __init__(self, num_reasoning_tokens=128, hidden_size=1024):
        super(ReasoningNet, self).__init__()
        
        self.reasoning_tokens = nn.Parameter(
            torch.randn(num_reasoning_tokens, hidden_size),
            requires_grad=True
        )

        self.num_reasoning_tokens = num_reasoning_tokens
        self.hidden_size = hidden_size


class MLPReasoningNet(ReasoningNet):
    def __init__(self, num_reasoning_tokens=128, hidden_size=1024, latent_size=512):
        """Initialize the MLP Reasoning Network."""
        super(MLPReasoningNet, self).__init__(num_reasoning_tokens, hidden_size)

        self.reasoning_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, latent_size),  
            torch.nn.Tanh(),
            torch.nn.Linear(latent_size, hidden_size)
        )

    def forward(self, hidden_states):
        """Forward pass through the MLP reasoning network."""
        # hidden_states: [batch_size, seq_length, hidden_size], left padding
        if hidden_states.dim() != 3:
            raise ValueError("hidden_states must be a 3D tensor with shape [batch_size, seq_length, hidden_size]")
    
        hidden_states = hidden_states[:, -1, :] # Get the last hidden state for each sequence in the batch
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.num_reasoning_tokens, 1) # Repeat for each soft prompt [batch_size, num_reasoning_tokens, hidden_size]
        reasoning_tokens = self.reasoning_tokens.unsqueeze(0).repeat(hidden_states.size(0), 1, 1) # [batch_size, num_reasoning_tokens, hidden_size]
        mix_embeddings = hidden_states * reasoning_tokens # Element-wise multiplication [batch_size, num_reasoning_tokens, hidden_size]
        mix_embeddings = mix_embeddings.to(self.reasoning_mlp[0].weight.dtype)
        reasoning_embeddings = self.reasoning_mlp(mix_embeddings)  # [batch_size, num_reasoning_tokens, hidden_size]
        reasoning_embeddings = reasoning_embeddings.to(hidden_states.dtype)
        return reasoning_embeddings
    

class TransformerReasoningNet(ReasoningNet):
    def __init__(self, model_name_or_path, num_reasoning_tokens=128, hidden_size=1024):
        """Initialize the Transformer Reasoning Network."""
        super(TransformerReasoningNet, self).__init__(num_reasoning_tokens, hidden_size)
        
        self.transformer = AutoModel.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,  # Use float16 for efficiency
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        self.transformer.embed_tokens = None # Remove the embedding layer

        if self.transformer.config.hidden_size != self.hidden_size:
            self.transform_layer = torch.nn.Linear(self.hidden_size, self.transformer.config.hidden_size)
            self.reverse_transform_layer = torch.nn.Linear(self.transformer.config.hidden_size, self.hidden_size)
            self.transform_layer.to(self.transformer.dtype)
            self.reverse_transform_layer.to(self.transformer.dtype)
        else:
            self.transform_layer = torch.nn.Identity()
            self.reverse_transform_layer = torch.nn.Identity()

    def forward(self, hidden_states):
        """Forward pass through the Transformer reasoning network."""
        if hidden_states.dim() != 3:
            raise ValueError("hidden_states must be a 3D tensor with shape [batch_size, seq_length, hidden_size]")
        
        hidden_states = hidden_states[:, -1, :]
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.num_reasoning_tokens, 1)
        
        # Repeat reasoning prompt for each batch
        reasoning_tokens = self.reasoning_tokens.unsqueeze(0).repeat(hidden_states.size(0), 1, 1)

        mix_embeddings = (hidden_states * reasoning_tokens).to(self.transformer.dtype) # Element-wise multiplication [batch_size, num_reasoning_tokens, hidden_size]
        
        # Dimension Transform
        mix_embeddings = self.transform_layer(mix_embeddings)

        # Attention mask is not needed
        outputs = self.transformer(inputs_embeds=mix_embeddings, output_hidden_states=True)
        reasoning_embeddings = outputs.hidden_states[-1]  # [batch_size, num_reasoning_tokens, hidden_size]

        # Revert the dimension transform
        reasoning_embeddings = self.reverse_transform_layer(reasoning_embeddings).to(hidden_states.dtype)
        
        return reasoning_embeddings 