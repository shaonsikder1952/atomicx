"""Graph Neural Network for Market Structure Analysis.

Models cross-asset relationships and contagion effects using Graph Attention Networks.
Captures how influence propagates through the market network (BTC→ETH→SPY→GOLD→DXY).

Uses PyTorch Geometric for efficient graph operations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from loguru import logger


@dataclass
class MarketGraph:
    """Market network representation."""
    assets: List[str]  # ["BTC/USDT", "ETH/USDT", "SPY", "GOLD", "DXY"]
    node_features: np.ndarray  # (num_assets, feature_dim)
    edge_index: np.ndarray  # (2, num_edges) - connectivity
    edge_weights: np.ndarray  # (num_edges,) - correlation strengths


@dataclass
class GraphPrediction:
    """GNN prediction output."""
    asset: str
    direction: str
    confidence: float
    probabilities: Dict[str, float]
    influence_from: Dict[str, float]  # Which assets influenced this prediction
    centrality_score: float  # How central this asset is in the network


class GraphAttentionLayer(nn.Module):
    """Graph Attention Network layer (GAT)."""

    def __init__(self, in_features: int, out_features: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features

        # Multi-head attention
        self.W = nn.Linear(in_features, n_heads * out_features, bias=False)
        self.a = nn.Parameter(torch.randn(n_heads, 2 * out_features))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, in_features)
            edge_index: Edge connectivity (2, num_edges)
        Returns:
            Updated node features (num_nodes, n_heads * out_features)
        """
        num_nodes = x.size(0)

        # Linear transformation
        Wh = self.W(x)  # (num_nodes, n_heads * out_features)
        Wh = Wh.view(num_nodes, self.n_heads, self.out_features)

        # Compute attention coefficients
        edge_src, edge_dst = edge_index[0], edge_index[1]

        # Concatenate source and destination features
        Wh_src = Wh[edge_src]  # (num_edges, n_heads, out_features)
        Wh_dst = Wh[edge_dst]  # (num_edges, n_heads, out_features)
        Wh_concat = torch.cat([Wh_src, Wh_dst], dim=-1)  # (num_edges, n_heads, 2*out_features)

        # Attention scores
        e = (Wh_concat * self.a).sum(dim=-1)  # (num_edges, n_heads)
        e = self.leakyrelu(e)

        # Normalize attention coefficients with softmax per node
        attention = torch.zeros(num_nodes, self.n_heads, device=x.device)
        attention = attention.index_add(0, edge_dst, torch.exp(e))
        attention = torch.exp(e) / (attention[edge_dst] + 1e-10)

        # Apply dropout
        attention = self.dropout(attention)

        # Aggregate neighbor features
        h_prime = torch.zeros(num_nodes, self.n_heads, self.out_features, device=x.device)
        for i in range(len(edge_src)):
            src, dst = edge_src[i], edge_dst[i]
            h_prime[dst] += attention[i].unsqueeze(-1) * Wh[src]

        # Concatenate heads
        h_prime = h_prime.view(num_nodes, self.n_heads * self.out_features)

        return h_prime


class MarketGNN(nn.Module):
    """Graph Neural Network for market structure prediction."""

    def __init__(
        self,
        input_dim: int = 46,  # 46 variables per asset
        hidden_dim: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(
                in_features=hidden_dim if i > 0 else hidden_dim,
                out_features=hidden_dim // n_heads,
                n_heads=n_heads,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

        # Output projection (per-node prediction)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # 3 classes: bullish, neutral, bearish
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge connectivity (2, num_edges)
        Returns:
            Predictions for each node (num_nodes, 3)
        """
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)

        # GAT layers
        for gat_layer in self.gat_layers:
            h = gat_layer(h, edge_index)
            h = F.relu(h)

        # Output projection
        out = self.output_proj(h)

        return out


class MarketGNNPredictor:
    """Market structure predictor using Graph Neural Networks.

    Models the market as a graph where:
    - Nodes = Assets (BTC, ETH, SPY, GOLD, DXY, OIL)
    - Edges = Correlations (dynamic, updated hourly)
    - Features = 46 variables per asset

    Learns how influence propagates through the network.

    Usage:
        predictor = MarketGNNPredictor()

        # Build market graph
        graph = predictor.build_graph({
            "BTC/USDT": btc_variables,
            "ETH/USDT": eth_variables,
            "SPY": spy_variables,
            "GOLD": gold_variables,
        })

        # Train on historical graphs
        predictor.train(historical_graphs, epochs=50)

        # Predict
        predictions = predictor.predict(current_graph)
        btc_pred = predictions["BTC/USDT"]
        print(f"BTC: {btc_pred.direction} (influenced by: {btc_pred.influence_from})")
    """

    def __init__(
        self,
        input_dim: int = 46,
        hidden_dim: int = 128,
        n_heads: int = 4,
        device: str = "mps",
    ):
        self.device = device
        self.input_dim = input_dim

        # Create model
        self.model = MarketGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
        ).to(device)

        self.asset_list = []  # Track asset order
        self.is_trained = False

        logger.info(f"[GNN] Initialized Graph Neural Network on {device}")

    def build_graph(
        self,
        asset_variables: Dict[str, Dict[str, float]],
        correlation_threshold: float = 0.3,
    ) -> MarketGraph:
        """Build market graph from asset variables.

        Args:
            asset_variables: {asset_name: {var_name: value}}
            correlation_threshold: Minimum correlation to create edge

        Returns:
            MarketGraph representation
        """
        assets = list(asset_variables.keys())
        num_assets = len(assets)

        # Build node features matrix
        node_features = np.zeros((num_assets, self.input_dim))
        for i, asset in enumerate(assets):
            variables = asset_variables[asset]
            # Convert dict to array (assuming variables are ordered)
            feat_vector = np.array([variables.get(f"VAR_{j}", 0.0) for j in range(self.input_dim)])
            node_features[i] = feat_vector

        # Build edge index (all-to-all for now, could filter by correlation)
        edge_list = []
        edge_weights = []

        for i in range(num_assets):
            for j in range(num_assets):
                if i != j:
                    # Calculate correlation between assets (using features)
                    corr = np.corrcoef(node_features[i], node_features[j])[0, 1]

                    if abs(corr) > correlation_threshold:
                        edge_list.append([i, j])
                        edge_weights.append(abs(corr))

        edge_index = np.array(edge_list).T if edge_list else np.array([[], []])
        edge_weights = np.array(edge_weights) if edge_weights else np.array([])

        graph = MarketGraph(
            assets=assets,
            node_features=node_features,
            edge_index=edge_index,
            edge_weights=edge_weights,
        )

        logger.debug(f"[GNN] Built graph: {num_assets} assets, {len(edge_list)} edges")
        return graph

    def train(
        self,
        graphs: List[MarketGraph],
        labels: List[np.ndarray],
        epochs: int = 50,
        learning_rate: float = 1e-3,
    ):
        """Train GNN on historical market graphs.

        Args:
            graphs: List of historical market graphs
            labels: List of ground truth labels (num_assets, 3) per graph
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        logger.info(f"[GNN] Training on {len(graphs)} historical graphs...")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0

            for graph, label in zip(graphs, labels):
                # Convert to tensors
                x = torch.FloatTensor(graph.node_features).to(self.device)
                edge_index = torch.LongTensor(graph.edge_index).to(self.device)
                y = torch.LongTensor(label).to(self.device)

                # Forward pass
                pred = self.model(x, edge_index)
                loss = F.cross_entropy(pred, y)

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(graphs)
            if (epoch + 1) % 10 == 0:
                logger.info(f"[GNN] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        self.is_trained = True
        logger.success(f"[GNN] Training complete!")

    @torch.no_grad()
    def predict(self, graph: MarketGraph) -> Dict[str, GraphPrediction]:
        """Predict for all assets in the graph.

        Args:
            graph: Current market graph

        Returns:
            Dictionary of predictions per asset
        """
        if not self.is_trained:
            logger.warning("[GNN] Model not trained, using baseline")
            return self._baseline_predictions(graph)

        self.model.eval()

        # Convert to tensors
        x = torch.FloatTensor(graph.node_features).to(self.device)
        edge_index = torch.LongTensor(graph.edge_index).to(self.device)

        # Forward pass
        logits = self.model(x, edge_index)
        probs = F.softmax(logits, dim=1).cpu().numpy()

        # Build predictions per asset
        predictions = {}
        classes = ["bearish", "neutral", "bullish"]

        for i, asset in enumerate(graph.assets):
            asset_probs = probs[i]
            predicted_class = classes[asset_probs.argmax()]
            confidence = float(asset_probs.max())

            # Calculate influence from neighbors
            influence_from = {}
            if len(graph.edge_index) > 0:
                # Find incoming edges to this node
                incoming_edges = graph.edge_index[1] == i
                source_nodes = graph.edge_index[0][incoming_edges]

                for src_idx in source_nodes:
                    src_asset = graph.assets[int(src_idx)]
                    # Influence = edge weight * source confidence
                    edge_idx = np.where((graph.edge_index[0] == src_idx) & (graph.edge_index[1] == i))[0]
                    if len(edge_idx) > 0:
                        weight = graph.edge_weights[int(edge_idx[0])]
                        influence_from[src_asset] = float(weight)

            # Calculate centrality (degree)
            degree = np.sum(graph.edge_index[1] == i) + np.sum(graph.edge_index[0] == i)
            centrality = float(degree) / len(graph.assets) if len(graph.assets) > 1 else 0.0

            predictions[asset] = GraphPrediction(
                asset=asset,
                direction=predicted_class,
                confidence=confidence,
                probabilities={
                    "bearish": float(asset_probs[0]),
                    "neutral": float(asset_probs[1]),
                    "bullish": float(asset_probs[2]),
                },
                influence_from=influence_from,
                centrality_score=centrality,
            )

        logger.info(f"[GNN] Predicted for {len(predictions)} assets")
        return predictions

    def _baseline_predictions(self, graph: MarketGraph) -> Dict[str, GraphPrediction]:
        """Simple baseline when model not trained."""
        predictions = {}

        for i, asset in enumerate(graph.assets):
            predictions[asset] = GraphPrediction(
                asset=asset,
                direction="neutral",
                confidence=0.5,
                probabilities={"bearish": 0.2, "neutral": 0.6, "bullish": 0.2},
                influence_from={},
                centrality_score=0.5,
            )

        return predictions

    def save(self, path: str):
        """Save trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'asset_list': self.asset_list,
        }, path)
        logger.info(f"[GNN] Model saved to {path}")

    def load(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.asset_list = checkpoint['asset_list']
        self.is_trained = True
        logger.info(f"[GNN] Model loaded from {path}")
