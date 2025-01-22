# Federated Learning with Adaptive Clustering

A modular implementation of federated learning with dynamic clustering strategies for client grouping and optimization.

## Features

- Multiple clustering algorithms (K-means, Hierarchical, Spectral, GMM)
- Adaptive cluster number selection
- Dynamic client grouping based on model weights, data features, or label distribution
- Automated experiment comparison
- Comprehensive metrics collection and visualization

## Requirements

- Python 3.8+
- PyTorch 2.0+
- scikit-learn 1.3.2
- numpy
- scipy 1.11.4

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Unix/MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
federated_clustering/
├── config/
│   └── config.yaml           # Configuration file
├── src/
│   ├── clustering.py         # Clustering implementations
│   ├── data.py              # Data management
│   ├── federated.py         # FL client/server
│   ├── metrics.py           # Performance metrics
│   ├── models.py            # Neural networks
│   └── utils.py             # Utilities
├── results/                  # Experiment results
├── main.py                  # Single experiment runner
└── run_clustering_comparison.py  # Multiple experiment runner
```

## Usage

### Single Experiment

Run a single federated learning experiment:

```bash
python main.py --config config/config.yaml --experiment_name test_run
```

### Clustering Comparison

Run multiple experiments comparing different clustering configurations:

```bash
python run_clustering_comparison.py
```

## Configuration

Key configuration parameters in `config.yaml`:

```yaml
data:
  dataset_name: "mnist"
  num_clients: 50
  iid: false

clustering:
  algorithm: "kmeans"  # kmeans, hierarchical, spectral, gmm
  clustering_type: "model_weights"  # model_weights, label_distribution, data_features
  pca_components: 5
  update_frequency: 5

fed:
  num_rounds: 10
  local_epochs: 5
  client_sample_ratio: 0.2
```

## Results Structure

Experiment results are saved in `clustering_experiments/`:

```
clustering_experiments/
├── comparison_{timestamp}/
│   ├── configs/            # Experiment configurations
│   ├── results/           # Individual results
│   │   ├── kmeans_1/
│   │   ├── hierarchical_1/
│   │   └── ...
│   ├── logs/             # Experiment logs
│   ├── results.json      # Combined results
│   └── summary.json      # Experiment summary
```

## Visualization

Results can be analyzed using the included visualization tools:

```python
from src.utils import plot_clustering_comparison
plot_clustering_comparison("clustering_experiments/comparison_latest")
```

## Citation

```bibtex

```

## License

MIT License - see LICENSE file for details.
