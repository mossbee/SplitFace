from torch import nn

class RelationEncoding(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers):
        super(RelationEncoding, self).__init__()
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, selected_tokens):
        # Apply attention layers
        for attention in self.attention_layers:
            selected_tokens, _ = attention(selected_tokens, selected_tokens, selected_tokens)

        # Process the output through a fully connected layer
        output = self.fc(selected_tokens)
        output = self.layer_norm(output)

        return output