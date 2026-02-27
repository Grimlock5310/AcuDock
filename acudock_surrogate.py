"""
AcuDock Surrogate Model - ML surrogate for active learning virtual screening.

Provides a SurrogateModel class that trains on Morgan fingerprints + molecular
descriptors to predict Vina docking scores. Supports uncertainty estimation
via Random Forest variance for acquisition function guidance.

Based on the HASTEN approach: >90% of top hits found with <10% docking effort.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class SurrogateModel:
    """ML surrogate model for predicting docking scores.

    Uses Morgan fingerprints + RDKit descriptors as features, and
    Random Forest regression for prediction with built-in uncertainty
    estimation (tree variance).
    """

    def __init__(self, fp_radius=2, fp_bits=2048, n_estimators=200,
                 random_state=42):
        """Initialize surrogate model.

        Args:
            fp_radius: Morgan fingerprint radius (default 2 = ECFP4).
            fp_bits: Fingerprint bit vector length.
            n_estimators: Number of trees in Random Forest.
            random_state: Seed for reproducibility.
        """
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.training_scores = []
        self.cv_scores = []

    def _smiles_to_features(self, smiles_list):
        """Convert SMILES list to feature matrix (fingerprints + descriptors).

        Returns (feature_matrix, valid_indices) where valid_indices tracks
        which SMILES were successfully featurized.
        """
        features = []
        valid_idx = []

        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            # Morgan fingerprint (ECFP)
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.fp_radius, nBits=self.fp_bits
            )
            fp_array = np.array(fp)

            # Molecular descriptors
            desc = np.array([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.RingCount(mol),
                Descriptors.FractionCSP3(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.HeavyAtomCount(mol),
            ])

            features.append(np.concatenate([fp_array, desc]))
            valid_idx.append(i)

        if not features:
            return np.array([]), []

        return np.array(features), valid_idx

    def train(self, smiles_list, scores):
        """Train the surrogate on (SMILES, docking_score) pairs.

        Args:
            smiles_list: List of SMILES strings.
            scores: Array of docking scores (kcal/mol, more negative = better).

        Returns dict with training metrics.
        """
        X, valid_idx = self._smiles_to_features(smiles_list)
        if len(X) == 0:
            raise ValueError('No valid molecules to train on.')

        y = np.array(scores)[valid_idx]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        # Cross-validation
        cv = cross_val_score(self.model, X_scaled, y, cv=min(5, len(y)), scoring='r2')
        self.cv_scores.append(cv.mean())

        # Training metrics
        y_pred = self.model.predict(X_scaled)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        r2 = self.model.score(X_scaled, y)

        metrics = {
            'n_samples': len(y),
            'n_features': X.shape[1],
            'train_r2': round(r2, 4),
            'train_rmse': round(rmse, 4),
            'cv_r2_mean': round(cv.mean(), 4),
            'cv_r2_std': round(cv.std(), 4),
        }
        self.training_scores.append(metrics)

        return metrics

    def predict(self, smiles_list):
        """Predict docking scores for new SMILES.

        Returns (predictions, valid_indices).
        """
        if not self.is_fitted:
            raise RuntimeError('Model not trained yet. Call train() first.')

        X, valid_idx = self._smiles_to_features(smiles_list)
        if len(X) == 0:
            return np.array([]), []

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        return predictions, valid_idx

    def get_uncertainty(self, smiles_list):
        """Estimate prediction uncertainty using tree variance.

        Each tree in the Random Forest makes an independent prediction.
        The standard deviation across trees serves as an uncertainty estimate.

        Returns (uncertainties, valid_indices).
        """
        if not self.is_fitted:
            raise RuntimeError('Model not trained yet. Call train() first.')

        X, valid_idx = self._smiles_to_features(smiles_list)
        if len(X) == 0:
            return np.array([]), []

        X_scaled = self.scaler.transform(X)

        # Get predictions from each tree
        tree_predictions = np.array([
            tree.predict(X_scaled) for tree in self.model.estimators_
        ])
        uncertainties = tree_predictions.std(axis=0)

        return uncertainties, valid_idx

    def acquisition_function(self, smiles_list, beta=1.0):
        """Score compounds for selection using Upper Confidence Bound (UCB).

        acquisition = -(predicted_score) + beta * uncertainty

        More negative predicted scores are better (stronger binding),
        so we negate to make lower scores -> higher acquisition.
        Higher uncertainty also increases acquisition value (exploration).

        Args:
            smiles_list: SMILES to evaluate.
            beta: Exploration-exploitation tradeoff (higher = more exploration).

        Returns (acquisition_scores, valid_indices).
        """
        predictions, valid_idx = self.predict(smiles_list)
        if len(predictions) == 0:
            return np.array([]), []

        uncertainties, _ = self.get_uncertainty(
            [smiles_list[i] for i in valid_idx]
        )

        # UCB: exploit strong binders + explore uncertain regions
        acquisition = -predictions + beta * uncertainties

        return acquisition, valid_idx

    def select_next_batch(self, candidate_smiles, batch_size=500, beta=1.0):
        """Select the next batch of compounds to dock.

        Uses the acquisition function to rank candidates, then selects
        the top batch_size compounds.

        Args:
            candidate_smiles: List of un-docked SMILES.
            batch_size: Number of compounds to select.
            beta: UCB exploration parameter.

        Returns list of (index, smiles, acquisition_score) tuples.
        """
        acq_scores, valid_idx = self.acquisition_function(candidate_smiles, beta=beta)

        if len(acq_scores) == 0:
            return []

        # Rank by acquisition score (descending)
        ranked = sorted(
            zip(valid_idx, acq_scores),
            key=lambda x: x[1],
            reverse=True
        )

        selected = []
        for idx, score in ranked[:batch_size]:
            selected.append((idx, candidate_smiles[idx], score))

        return selected

    def get_feature_importance(self, top_n=20):
        """Get top feature importances from the Random Forest.

        Returns DataFrame with feature names and importance scores.
        """
        if not self.is_fitted:
            return pd.DataFrame()

        importances = self.model.feature_importances_

        # Name the features
        fp_names = [f'FP_bit_{i}' for i in range(self.fp_bits)]
        desc_names = [
            'MW', 'LogP', 'TPSA', 'HBD', 'HBA',
            'RotBonds', 'RingCount', 'FractionCSP3',
            'AromaticRings', 'HeavyAtomCount'
        ]
        all_names = fp_names + desc_names

        df = pd.DataFrame({
            'Feature': all_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(top_n)

        return df.reset_index(drop=True)
