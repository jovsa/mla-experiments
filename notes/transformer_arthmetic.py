class TransformerInference:
    def __init__(self, config, hardware):
        self.config = config
        self.hardware = hardware

        # Validate configuration
        assert self.config['d_ff'] == 4 * self.config['d_model'], "d_ff should be 4*d_model"
        assert self.config['d_model'] % self.config['n_heads'] == 0, "d_model must be divisible by n_heads"

    def total_parameters(self):
        """Calculate total number of parameters in the transformer"""
        # Embedding parameters
        embed_params = self.config['vocab_size'] * self.config['d_model']

        # Transformer block parameters (per layer)
        attn_params = 4 * self.config['d_model']**2  # Q,K,V + output projection
        ff_params = 2 * self.config['d_model'] * self.config['d_ff']  # Two linear layers
        layer_params = attn_params + ff_params + 2*self.config['d_model']  # + layer norm

        # Total parameters
        return embed_params + self.config['n_layers'] * layer_params

    def flops_per_token(self):
        """Calculate FLOPs required for processing one token"""
        # Attention operations
        attn_flops = (
            2 * self.config['d_model']**2 +  # Q,K,V projections
            self.config['d_model']**2 +      # Output projection
            2 * self.config['d_model'] * self.config['context_length']  # Attention computation
        )

        # Feed-forward operations
        ff_flops = 2 * self.config['d_ff'] * self.config['d_model']

        # Total per layer
        layer_flops = attn_flops + ff_flops

        # Total for all layers + embeddings
        return self.config['n_layers'] * layer_flops + 2 * self.config['d_model'] * self.config['vocab_size']

    def memory_bandwidth(self):
        """Calculate memory bandwidth required for inference"""
        return self.total_parameters() * self.hardware['bytes_per_param']

    def latency(self, batch_size=1):
        """Calculate theoretical latency for processing one token"""
        # Compute-bound latency
        total_flops = self.flops_per_token() * batch_size
        compute_time = total_flops / self.hardware['flops']

        # Memory-bound latency
        memory_time = self.memory_bandwidth() / self.hardware['memory_bandwidth']

        return max(compute_time, memory_time)

    def throughput(self, batch_size=1):
        """Calculate maximum theoretical tokens per second"""
        return batch_size / self.latency(batch_size)

# Example usage
config = {
    'n_layers': 12,
    'd_model': 768,
    'n_heads': 12,
    'd_ff': 3072,
    'vocab_size': 50257,
    'context_length': 2048
}

hardware = {
    'flops': 312e12,  # A100 GPU FP16 performance
    'memory_bandwidth': 1.5e12,  # A100 memory bandwidth in bytes/s
    'bytes_per_param': 2  # FP16
}

model = TransformerInference(config, hardware)

print(f"Total Parameters: {model.total_parameters()/1e6:.1f}M")
print(f"FLOPs per token: {model.flops_per_token()/1e9:.1f} GFLOPs")
print(f"Theoretical Latency: {model.latency()*1e3:.2f}ms per token")
print(f"Theoretical Throughput: {model.throughput(128):.0f} tokens/sec (bs=128)")