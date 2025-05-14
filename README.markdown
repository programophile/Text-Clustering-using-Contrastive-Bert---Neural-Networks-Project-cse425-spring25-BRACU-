# TweetEval Emotion Clustering with BERT and Contrastive Learning

This project implements a pipeline for clustering tweets from the TweetEval dataset (emotion subset) using a custom BERT-based encoder with a contrastive learning approach. The model fine-tunes a BERT encoder with a projection head, embeds the dataset, applies KMeans clustering, evaluates clustering performance, and visualizes the results using t-SNE.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
The project processes the TweetEval emotion dataset (up to 10,000 samples) to:
1. Prepare positive and negative text pairs for contrastive learning.
2. Fine-tune a custom BERT-based encoder with a projection head.
3. Generate 128-dimensional embeddings for the dataset.
4. Cluster the embeddings using KMeans (4 clusters).
5. Evaluate clustering with Silhouette and Davies-Bouldin scores.
6. Visualize clusters using t-SNE and a scatter plot.

The pipeline leverages PyTorch, Hugging Face Transformers, scikit-learn, and visualization libraries to achieve these tasks.

## Features
- **Contrastive Learning**: Uses a custom contrastive loss to fine-tune BERT embeddings.
- **Custom BERT Encoder**: Combines `bert-base-uncased` with a projection head (768→512→128 dimensions).
- **Clustering**: Applies KMeans to group similar tweets.
- **Evaluation**: Computes Silhouette and Davies-Bouldin scores for clustering quality.
- **Visualization**: Generates a t-SNE scatter plot to visualize clusters.

## Requirements
- Python 3.8+
- Libraries:
  - `torch>=1.9.0`
  - `transformers>=4.20.0`
  - `datasets>=2.0.0`
  - `scikit-learn>=1.0.0`
  - `numpy>=1.19.0`
  - `matplotlib>=3.4.0`
  - `seaborn>=0.11.0`
- CUDA-enabled GPU (optional, for faster training)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/tweeteval-emotion-clustering.git
   cd tweeteval-emotion-clustering
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install torch transformers datasets scikit-learn numpy matplotlib seaborn
   ```
4. Ensure CUDA is set up if using a GPU (see [PyTorch installation guide](https://pytorch.org/get-started/locally/)).

## Usage
1. Run the main script to execute the full pipeline:
   ```bash
   python main.py
   ```
2. The script will:
   - Load the TweetEval dataset.
   - Prepare text pairs and fine-tune the BERT encoder.
   - Generate embeddings and cluster them.
   - Output clustering metrics and save a t-SNE visualization (`tsne_plot.png`).

3. Example output:
   ```
   🔹 Loading TweetEval data...
   🔹 Preparing pairs...
   🔹 Fine-tuning custom BERT encoder...
   Epoch 1 | Loss: 0.XXXX
   ...
   🔹 Clustering embeddings...
   Cluster distribution: (array([0, 1, 2, 3]), array([XXXX, XXXX, XXXX, XXXX]))
   🔹 Evaluating clustering...
   Silhouette Score: X.XXXX
   Davies-Bouldin Score: X.XXXX
   🔹 Visualizing with t-SNE...
   ```

4. The t-SNE plot will be saved as `tsne_plot.png` in the project directory.

## File Structure
```
tweeteval-emotion-clustering/
├── main.py                 # Main script with the full pipeline
├── README.md               # Project documentation
├── tsne_plot.png           # Output t-SNE visualization (generated after running)
└── requirements.txt        # List of dependencies
```

## Methodology
1. **Data Loading**: Loads the TweetEval emotion dataset (train split, 10,000 samples) using the `datasets` library.
2. **Pair Preparation**: Creates positive (same label) and negative (different labels) text pairs for contrastive learning.
3. **Tokenization**: Uses `BertTokenizer` (`bert-base-uncased`) to tokenize texts with a max length of 128.
4. **Model**: A custom `CustomBERTEncoder` with:
   - Pre-trained `BertModel` (`bert-base-uncased`).
   - Projection head: Linear (768→512), ReLU, Dropout (0.3), Linear (512→128).
5. **Training**: Fine-tunes the model for 5 epochs using contrastive loss and Adam optimizer (lr=1e-5).
6. **Embedding**: Generates 128-dimensional embeddings for the entire dataset.
7. **Clustering**: Applies KMeans (n_clusters=4) to the embeddings.
8. **Evaluation**: Computes Silhouette and Davies-Bouldin scores.
9. **Visualization**: Reduces embeddings to 2D with t-SNE and plots them with Seaborn.

## Results
- **Cluster Distribution**: Varies based on data; typically balanced across 4 clusters.
- **Metrics**: Silhouette and Davies-Bouldin scores indicate clustering quality.
- **Visualization**: A t-SNE scatter plot (`tsne_plot.png`) shows clusters in 2D space.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.