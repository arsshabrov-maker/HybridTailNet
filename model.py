import numpy as np
import torch
import torch.nn as nn


def create_topology_blueprint(hidden_dims, non_neuron_prob=0.35, seed=42):
  """
    Generates a fixed random placement of Standard Neurons (0) and Non-Neurons (1).
    Ensures at least one Non-Neuron and one Standard Neuron per hidden layer.
    """
  rng = np.random.RandomState(seed)
  blueprint = []
  for dim in hidden_dims:
    mask = (rng.rand(dim) < non_neuron_prob).astype(np.float32)
    mask[0] = 1.0
    mask[-1] = 0.0
    blueprint.append(torch.tensor(mask))
  return blueprint


class SharedNonNeuron(nn.Module):
  """
    Shared Non-Neuron unit (Global Context Compressor).
    Reads a 10-dimensional state vector:
      [node_activation (1) + layer_stats: mean, max, min (3) + zero-padded tail (6)]
    and outputs a context-modulated scalar.
    """

  def __init__(self, max_len=10, mini_hidden=12):
    super().__init__()
    self.max_len = max_len
    self.mini_mlp = nn.Sequential(
        nn.Linear(max_len, mini_hidden),
        nn.GELU(),
        nn.Linear(mini_hidden, 1),
    )

  def forward(self, node_vals, tail):
    batch_size, num_nodes = node_vals.shape

    # Layer-level summary statistics
    l_mean = node_vals.mean(dim=1, keepdim=True)
    l_max = node_vals.max(dim=1, keepdim=True).values
    l_min = node_vals.min(dim=1, keepdim=True).values
    stats = torch.cat([l_mean, l_max, l_min], dim=1)

    # Zero-pad topological tail up to (max_len - 4) slots and log-scale counts
    tail_slots = self.max_len - 4
    padded_tail = torch.zeros(batch_size, tail_slots, device=node_vals.device)
    valid_len = min(tail.size(1), tail_slots)
    padded_tail[:, :valid_len] = (
        torch.log1p(torch.clamp(tail[:, :valid_len], min=0.0)) * 0.25
    )

    # Broadcast context across all nodes in the layer
    context = torch.cat([stats, padded_tail], dim=1)
    context_expanded = context.unsqueeze(1).expand(
        batch_size, num_nodes, self.max_len - 1
    )

    inp = torch.cat([node_vals.unsqueeze(-1), context_expanded], dim=-1)
    return self.mini_mlp(inp).squeeze(-1)


class HybridTailNet(nn.Module):
  """
    HybridTailNet Architecture.
    Combines standard neuron accumulators with shared Non-Neuron context compressors
    and propagates a dynamically growing topological metadata tail across layers.
    """

  def __init__(
      self,
      in_dim,
      hidden_dims=(24, 24, 16),
      non_neuron_prob=0.35,
      max_len=10,
      mini_hidden=12,
      seed=42,
  ):
    super().__init__()
    self.hidden_dims = list(hidden_dims)
    self.blueprint = create_topology_blueprint(
        self.hidden_dims, non_neuron_prob=non_neuron_prob, seed=seed
    )

    dims = [in_dim] + self.hidden_dims
    self.layers = nn.ModuleList(
        [nn.Linear(dims[i], dims[i + 1]) for i in range(len(self.hidden_dims))]
    )
    self.shared_non_neuron = SharedNonNeuron(
        max_len=max_len, mini_hidden=mini_hidden
    )
    self.head = nn.Linear(self.hidden_dims[-1], 1)

  def forward(self, x, init_mask=None):
    batch_size = x.size(0)
    if init_mask is not None:
      tail = init_mask.sum(dim=1, keepdim=True)
    else:
      tail = torch.full((batch_size, 1), float(x.size(1)), device=x.device)

    h = x
    for layer, mask in zip(self.layers, self.blueprint):
      mask = mask.to(x.device).unsqueeze(0)
      n_std = (1.0 - mask).sum()
      n_non = mask.sum()

      # 1. Standard neuron transformation
      h_std = torch.nn.functional.gelu(layer(h))

      # 2. Shared Non-Neuron context modulation
      h_non = self.shared_non_neuron(h_std, tail)

      # 3. Blend according to fixed topology blueprint
      h = h_std * (1.0 - mask) + h_non * mask

      # 4. Grow topological metadata tail and append depth marker 1.0
      ones = torch.ones(batch_size, 1, device=x.device)
      if n_non > 0:
        ratio_std = n_std / mask.size(1)
        tail = torch.cat(
            [tail * max(1.0, n_std.item()) * ratio_std, ones], dim=1
        )
      else:
        tail = torch.cat([tail * mask.size(1), ones], dim=1)

    return self.head(h)


class StandardMLP(nn.Module):
  """Parameter-matched standard MLP baseline."""

  def __init__(self, in_dim, hidden_dims=(28, 28, 20)):
    super().__init__()
    dims = [in_dim] + list(hidden_dims)
    layers = []
    for i in range(len(dims) - 1):
      layers.append(nn.Linear(dims[i], dims[i + 1]))
      layers.append(nn.GELU())
    layers.append(nn.Linear(dims[-1], 1))
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x)
