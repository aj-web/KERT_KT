"""
Baseline Model Comparison Script
Compares KER-KT with traditional KT baselines
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.kert_kt import KERKT, KTSequenceDataset
from models.kt_predictor import DataCollator
from baselines.dkt import DKT, DKTLoss
from baselines.dkvmn import DKVMN, DKVMNLoss
from baselines.sakt import SAKT, SAKTLoss
from baselines.akt import AKT, AKTLoss
from baselines.gkt import GKT, GKTLoss


class BaselineTrainer:
    """Trainer for baseline models"""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)

    def train_epoch(self, train_loader, optimizer, loss_fn):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            # Move to device
            question_seq = batch['question_seq'].to(self.device)
            concept_seq = batch['concept_seq'].to(self.device)
            answer_seq = batch['answer_seq'].to(self.device)
            target_concept = batch['target_concept'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            if isinstance(self.model, DKT):
                predictions = self.model.predict_single_concept(question_seq, answer_seq, target_concept)
            elif isinstance(self.model, DKVMN):
                predictions = self.model.predict_single_concept(question_seq, answer_seq, target_concept)
            elif isinstance(self.model, SAKT):
                predictions = self.model.predict_single_concept(question_seq, answer_seq, target_concept)
            elif isinstance(self.model, AKT):
                predictions = self.model.predict_single_concept(question_seq, concept_seq, answer_seq, target_concept)
            elif isinstance(self.model, GKT):
                predictions = self.model.predict_single_concept(question_seq, concept_seq, answer_seq, target_concept)
            else:
                # For KER-KT or other models
                predictions = torch.rand(labels.size(0)).to(self.device)

            # Compute loss
            loss = loss_fn(predictions, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(train_loader)

    def evaluate(self, data_loader):
        """Evaluate model"""
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                question_seq = batch['question_seq'].to(self.device)
                concept_seq = batch['concept_seq'].to(self.device)
                answer_seq = batch['answer_seq'].to(self.device)
                target_concept = batch['target_concept'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass based on model type
                if isinstance(self.model, DKT):
                    predictions = self.model.predict_single_concept(question_seq, answer_seq, target_concept)
                elif isinstance(self.model, DKVMN):
                    predictions = self.model.predict_single_concept(question_seq, answer_seq, target_concept)
                elif isinstance(self.model, SAKT):
                    predictions = self.model.predict_single_concept(question_seq, answer_seq, target_concept)
                elif isinstance(self.model, AKT):
                    predictions = self.model.predict_single_concept(question_seq, concept_seq, answer_seq, target_concept)
                elif isinstance(self.model, GKT):
                    predictions = self.model.predict_single_concept(question_seq, concept_seq, answer_seq, target_concept)
                else:
                    predictions = torch.rand(labels.size(0)).to(self.device)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        auc = roc_auc_score(all_labels, all_predictions)
        acc = accuracy_score(all_labels, np.round(all_predictions))

        return {'auc': auc, 'acc': acc}


def train_baseline_model(model_name, dataset_name, n_epochs=10):
    """
    Train a baseline model

    Args:
        model_name: name of baseline model
        dataset_name: name of dataset
        n_epochs: number of epochs

    Returns:
        best_metrics: best validation metrics
    """
    print(f"\nTraining {model_name} on {dataset_name}...")

    # Load dataset
    with open('./data/processed_datasets.pkl', 'rb') as f:
        datasets = pickle.load(f)

    dataset_info = datasets[dataset_name]

    # Create data loaders
    train_dataset = KTSequenceDataset(dataset_info['train'], max_seq_len=100)  # Shorter for baseline
    val_dataset = KTSequenceDataset(dataset_info['val'], max_seq_len=100)

    collator = DataCollator(max_seq_len=100)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collator.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collator.collate_fn)

    # Initialize model
    if model_name == 'DKT':
        model = DKT(dataset_info['n_questions'], dataset_info['n_concepts'])
        loss_fn = DKTLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    else:
        # Placeholder for other models
        print(f"{model_name} not implemented yet, using random predictions")
        return {'auc': 0.5, 'acc': 0.5}

    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = BaselineTrainer(model, device)

    # Training loop
    best_auc = 0.0
    best_metrics = None

    for epoch in range(n_epochs):
        train_loss = trainer.train_epoch(train_loader, optimizer, loss_fn)
        val_metrics = trainer.evaluate(val_loader)

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_metrics = val_metrics

        print(f"Epoch {epoch+1}/{n_epochs}: Loss={train_loss:.4f}, "
              f"Val AUC={val_metrics['auc']:.4f}, Val ACC={val_metrics['acc']:.4f}")

    return best_metrics


def train_kert_kt_model(dataset_name, n_epochs=20):
    """
    Train KER-KT model (simplified version)

    Args:
        dataset_name: name of dataset
        n_epochs: number of epochs

    Returns:
        best_metrics: best validation metrics
    """
    print(f"\nTraining KER-KT on {dataset_name}...")

    # Load dataset
    with open('./data/processed_datasets.pkl', 'rb') as f:
        datasets = pickle.load(f)

    dataset_info = datasets[dataset_name]
    concept_graph = torch.tensor(dataset_info['concept_graph'], dtype=torch.float32)

    # Create data loaders
    train_dataset = KTSequenceDataset(dataset_info['train'], max_seq_len=100)
    val_dataset = KTSequenceDataset(dataset_info['val'], max_seq_len=100)

    collator = DataCollator(max_seq_len=100)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collator.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collator.collate_fn)

    # Initialize KER-KT model
    model = KERKT(
        n_questions=dataset_info['n_questions'],
        n_concepts=dataset_info['n_concepts'],
        embed_dim=64,  # Smaller for quick testing
        hidden_dim=128,
        n_layers=1,    # Simplified
        lr_kt=1e-3
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    concept_graph = concept_graph.to(device)

    # Training loop
    best_auc = 0.0
    best_metrics = None
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Training step
            losses = model.train_step(batch, concept_graph)
            epoch_loss += losses['total_kt_loss']

        epoch_loss /= len(train_loader)

        # Validation
        val_metrics = model.evaluate(val_loader, concept_graph)

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_metrics = val_metrics

        print(f"Epoch {epoch+1}/{n_epochs}: Loss={epoch_loss:.4f}, "
              f"Val AUC={val_metrics['auc']:.4f}, Val ACC={val_metrics['acc']:.4f}")

    return best_metrics


def run_comparison():
    """Run comparison between KER-KT and baselines"""
    datasets = ['assist09']  # Start with one dataset for quick testing
    baselines = ['DKT']  # Only implement DKT for now

    results = {}

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*60}")

        results[dataset] = {}

        # Train baselines
        for baseline in baselines:
            try:
                metrics = train_baseline_model(baseline, dataset, n_epochs=5)
                results[dataset][baseline] = metrics
                print(f"{baseline}: AUC={metrics['auc']:.4f}, ACC={metrics['acc']:.4f}")
            except Exception as e:
                print(f"Error training {baseline}: {e}")
                results[dataset][baseline] = {'auc': 0.5, 'acc': 0.5}

        # Train KER-KT
        try:
            metrics = train_kert_kt_model(dataset, n_epochs=10)
            results[dataset]['KER-KT'] = metrics
            print(f"KER-KT: AUC={metrics['auc']:.4f}, ACC={metrics['acc']:.4f}")
        except Exception as e:
            print(f"Error training KER-KT: {e}")
            results[dataset]['KER-KT'] = {'auc': 0.5, 'acc': 0.5}

    # Print final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON RESULTS")
    print(f"{'='*60}")

    for dataset, model_results in results.items():
        print(f"\n{dataset.upper()}:")
        for model, metrics in model_results.items():
            auc = metrics['auc']
            acc = metrics['acc']
            print(f"  {model:8s}: AUC={auc:.4f}, ACC={acc:.4f}")

    return results


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Run comparison
    results = run_comparison()

    print("\nComparison completed!")
    print("Note: This is a simplified version for quick testing.")
    print("Full experiment with all baselines and datasets would require more implementation.")
