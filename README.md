# Jigsaw - Agile Community Rules Classification

A machine learning solution for predicting whether Reddit comments violate specific subreddit rules using transformer-based models and ensemble methods.

## 🏆 Competition Results

- **Rank**: 82nd out of 2,445 teams (Top 3.4%)
- **Medal**: Silver 🥈
- **Metric**: Column-averaged AUC
- **Competition Host**: Jigsaw/Conversation AI

## 📋 Problem Statement

This competition addresses the challenge of understanding subreddit-specific moderation. Each subreddit has unique guidelines, and determining whether a comment violates a specific rule requires understanding both the content and community context.

**Task**: Build a binary classifier that predicts whether a Reddit comment broke a specific rule from a given subreddit.

## 🎯 Solution Overview

Our solution employs an ensemble approach combining:
- **Multiple transformer models** (DeBERTa, BGE embeddings)
- **Triplet learning** for improved representation learning
- **Data augmentation** techniques
- **Multi-model ensemble** (4x14B + 2x7B + 2xTriplet models)

## 📁 Project Structure

```
├── config/                      # Configuration files
│   ├── config.yaml
├── src/jigsaw/                  # Main source code
│   ├── components/              # Core components
│   │   ├── data/               # Data processing
│   │   │   ├── augmentation/   # Text augmentation
│   │   │   ├── transformation/ # Data transformation
│   │   │   └── validation/     # Data validation
│   │   ├── dataset/            # Dataset classes
│   │   ├── engine/             # Training engine
│   │   ├── models/             # Model definitions
│   │   └── train.py            # Training script
│   ├── config/                 # Configuration management
│   ├── constants/              # Constants and prompts
│   ├── core/                   # Core entities
│   ├── pipelines/              # End-to-end pipelines
│   │   ├── data.py            # Data pipeline
│   │   ├── train.py           # Training pipeline
│   │   └── inference.py       # Inference pipeline
│   └── utils/                  # Utility functions
├── schemas/                    # YAML schemas
├── scripts/                    # Utility scripts
├── working/                    # Notebooks & experiments
├── dist/                       # Distribution files
├── main.py                     # Main entry point
└── docker-compose.yaml         # Docker configuration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or 3.13
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd jigsaw-competition
```

2. **Install dependencies**
```bash
pip install uv 
uv sync
```

3. **Set up configuration**
```bash
cp config/config.yaml config/config_local.yaml
# Edit config_local.yaml with your settings
```

### Running the Pipeline
```bash
python main.py --pipeline data
```

## 🔧 Configuration

Configuration files are located in the `config/` directory. Key parameters:
eight_decay: 0.01


## 📊 Model Architecture

### Base Models
- **Qwen2.5-14B**: model for better performance
- **Qwen2.5-7B**: For ensembling
- **Qwen3-14B**: Diversification
- **Qwen3-8B**: Diversification
- **DeBERTa-v3-small**: Fast and efficient transformer
- **BGE-base-en-v1.5**: Embedding model for semantic understanding
- **BGE-large-en-v1.5**: Larger embedding model for better performance

### Training Strategy
1. **Triplet Learning**: Learn better embeddings by comparing similar/dissimilar comments
2. **Cross-Validation**: 5-fold stratified cross-validation
3. **Data Augmentation**: Text augmentation for better generalization
4. **Ensemble**: Combine multiple models for robust predictions

## 🔬 Key Features

### Data Processing
- **Text Cleaning**: Remove noise, standardize formatting
- **Validation**: Ensure data quality
- **Augmentation**: Back-translation, synonym replacement
- **Zero-shot Learning**: Leverage pre-trained knowledge

### Model Training
- **Multi-GPU Support**: Distributed training capability
- **Mixed Precision**: FP16 training for faster computation
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Early Stopping**: Prevent overfitting

### Inference
- **Batch Processing**: Efficient prediction on large datasets
- **Model Ensemble**: Weighted averaging of predictions
- **TTA (Test Time Augmentation)**: Multiple predictions per sample

## 📝 Scripts

Useful scripts in `scripts/`:
- `clean.sh`: Clean up temporary files and caches
- `notebook.sh`: Launch Jupyter notebook server
- `publish_kaggle.sh`: Package and publish to Kaggle
- `push.sh`: Push code to repository

## 🧪 Notebooks

Experimental notebooks in `working/`:
- `training.ipynb`: Model training experiments
- `training_engine.ipynb`: Engine development
- `data_transformation.ipynb`: Data processing exploration
- `kaggle_runtime.ipynb`: Kaggle submission notebook
- `submission_runtime_v01.ipynb`: Final submission workflow

## 📚 References

- [Competition Page](https://kaggle.com/competitions/jigsaw-agile-community-rules)

## 📄 License

See `LICENSE` file for details.

## 🙏 Acknowledgments

- **Jigsaw/Conversation AI** for hosting the competition
- **Kaggle** community for discussions and insights
- Research by Deepak Kumar, Yousef AbuHashem, Zakir Durumeric
- Dataset work by Eshwar Chandrasekharan and Eric Gilbert

## 📧 Contact

For questions or collaboration:
- Kaggle: morizin
- GitHub: morizin

---

**Note**: This project was developed for the Kaggle competition "Jigsaw - Agile Community Rules Classification" and achieved a Silver Medal (82nd place out of 2,445 teams).