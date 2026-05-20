import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_regression


class SimpleModelEvaluator:
    """Simplified evaluator for internal use in FeatureAgent."""
    
    def __init__(self, model, eval_metric=roc_auc_score, random_state=42, col_id='client_id', target_col='bins', cols_budget=None):
        self.model = model
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.col_id = col_id
        self.target_col = target_col
        self.cols_budget = cols_budget

    def _prepare_features(self, train_features: pd.DataFrame, test_features: pd.DataFrame,
                         train_target: pd.Series, test_target: pd.Series):
        """Prepare and validate features for evaluation."""
        # Remove client_id and non-numeric columns
        train_features = train_features.drop(columns=[self.col_id, self.target_col], errors='ignore')
        test_features = test_features.drop(columns=[self.col_id, self.target_col], errors='ignore')
        
        # Select only numeric columns
        numeric_train = train_features.select_dtypes(include=[np.number])
        numeric_test = test_features.select_dtypes(include=[np.number])
        
        if len(numeric_train.columns) == 0 or len(numeric_test.columns) == 0:
            raise ValueError("No numeric features available for evaluation")
        
        # Ensure both train and test have same columns in same order
        common_cols = numeric_train.columns.intersection(numeric_test.columns)
        if len(common_cols) == 0:
            raise ValueError("No common columns between train and test")
            
        numeric_train = numeric_train[common_cols]
        numeric_test = numeric_test[common_cols]
        
        # Reset indices to ensure alignment
        numeric_train = numeric_train.sort_index(axis=0).reset_index(drop=True)
        numeric_test = numeric_test.sort_index(axis=0).reset_index(drop=True)
        train_target = train_target.sort_index(axis=0).reset_index(drop=True)
        test_target = test_target.sort_index(axis=0).reset_index(drop=True)
        
        # Verify shapes
        if len(numeric_train) != len(train_target):
            raise ValueError(f"Train features {len(numeric_train)} and target {len(train_target)} length mismatch")
        if len(numeric_test) != len(test_target):
            raise ValueError(f"Test features {len(numeric_test)} and target {len(test_target)} length mismatch")
        
        return numeric_train, numeric_test, train_target, test_target

    def _train_and_evaluate(self, train_features: pd.DataFrame, test_features: pd.DataFrame,
                           train_target: pd.Series, test_target: pd.Series) -> float:
        """Train model and evaluate on test set."""
        model = clone(self.model)
        model.fit(train_features, train_target)
        
        if hasattr(model, 'predict_proba') and self.eval_metric == roc_auc_score:
            probs = model.predict_proba(test_features)[:, 1]
            score = self.eval_metric(test_target, probs)
        else:
            preds = model.predict(test_features)
            score = self.eval_metric(test_target, preds)
        
        return score, model

    def _select_top_features(self, features_importances_df: pd.DataFrame, 
                            train_features: pd.DataFrame, test_features: pd.DataFrame,
                            n_features: int) -> tuple:
        """Select top N features based on importance."""
        top_cols = features_importances_df['feature'].head(n_features).tolist()
        train_selected = train_features[top_cols]
        test_selected = test_features[top_cols]
        return train_selected, test_selected, top_cols

    def evaluate_feature_set(self, train_features: pd.DataFrame, test_features: pd.DataFrame, 
                           train_target: pd.Series, test_target: pd.Series) -> tuple:
        """Properly evaluate feature set with validation checks."""
        try:
            # Prepare and validate features
            numeric_train, numeric_test, train_target, test_target = self._prepare_features(
                train_features, test_features, train_target, test_target
            )
            
            # Evaluate with all features
            score, model = self._train_and_evaluate(numeric_train, numeric_test, train_target, test_target)
            print(f"All features score: {score}")
            
            # Get feature importances
            features_importances_df = self.get_importances_df(model, numeric_train.columns)
            
            # Evaluate with top features if budget is specified
            if self.cols_budget is not None:
                train_selected, test_selected, top_cols = self._select_top_features(
                    features_importances_df, numeric_train, numeric_test, self.cols_budget
                )
                score, _ = self._train_and_evaluate(train_selected, test_selected, train_target, test_target)
                print(f"Top {self.cols_budget} features score: {score}")
                selected_cols = [self.col_id] + top_cols
            else:
                selected_cols = [self.col_id] + list(numeric_train.columns)
            
            return score, selected_cols, features_importances_df
            
        except Exception as e:
            warnings.warn(f"Error evaluating feature set: {e}")
            return 0.0, [], pd.DataFrame()

    def get_importances_df(self, model, cols):
        """Get importances dataframe"""
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = getattr(model, "feature_importances_", None)
        elif hasattr(model, "get_feature_importance"):  # CatBoost
            importances = model.get_feature_importance()
        if importances is None:
            if hasattr(model, "coef_"):
                importances = np.abs(model.coef_).flatten()
            else:
                raise ValueError("No importances found")
        return pd.DataFrame({
            'feature': cols,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

    def debug_evaluation(self, train_features: pd.DataFrame, test_features: pd.DataFrame, 
                    train_target: pd.Series, test_target: pd.Series):
        """Debug method to identify why scores are always 1.0"""
        
        print("=== DEBUG EVALUATION ===")
        
        # Check shapes
        print(f"Train features: {train_features.shape}, Train target: {train_target.shape}")
        print(f"Test features: {test_features.shape}, Test target: {test_target.shape}")
        
        # Check for target in features
        if any(col for col in train_features.columns if 'target' in str(col).lower()):
            print("⚠️ WARNING: Target column found in features!")
        
        # Check target distribution
        print(f"Train target unique values: {train_target.unique()}")
        print(f"Test target unique values: {test_target.unique()}")
        print(f"Train target value counts:\n{train_target.value_counts()}")
        print(f"Test target value counts:\n{test_target.value_counts()}")
        
        # Check if all predictions would be the same class
        if train_target.nunique() == 1:
            print("⚠️ WARNING: Train target has only one class!")
        if test_target.nunique() == 1:
            print("⚠️ WARNING: Test target has only one class!")
        
        # Sample predictions to see what's happening
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(train_features.drop(columns=[self.col_id], errors='ignore'), train_target)
        dummy_pred = dummy.predict(test_features.drop(columns=[self.col_id], errors='ignore'))
        dummy_score = self.eval_metric(test_target, dummy_pred)
        print(f"Dummy classifier score: {dummy_score}")
        
        print("=======================")


class SimpleModelInterpreter:
    """Interpreter for feature selection without target supervision."""
    
    def __init__(self, model=None, eval_metric=roc_auc_score, random_state=42, 
                 col_id='client_id', target_col='bins', cols_budget=None, 
                 downstream_mode=True):
        self.model = model
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.col_id = col_id
        self.target_col = target_col
        self.cols_budget = cols_budget
        self.downstream_mode = downstream_mode

    def _prepare_features(self, train_features: pd.DataFrame, test_features: pd.DataFrame,
                         train_target: pd.Series = None, test_target: pd.Series = None):
        """Prepare and validate features for interpretation."""
        # Remove client_id and non-numeric columns
        train_features = train_features.drop(columns=[self.col_id, self.target_col], errors='ignore')
        test_features = test_features.drop(columns=[self.col_id, self.target_col], errors='ignore')
        
        # Select only numeric columns
        numeric_train = train_features.select_dtypes(include=[np.number])
        numeric_test = test_features.select_dtypes(include=[np.number])
        
        if len(numeric_train.columns) == 0 or len(numeric_test.columns) == 0:
            raise ValueError("No numeric features available for interpretation")
        
        # Ensure both train and test have same columns in same order
        common_cols = numeric_train.columns.intersection(numeric_test.columns)
        if len(common_cols) == 0:
            raise ValueError("No common columns between train and test")
            
        numeric_train = numeric_train[common_cols]
        numeric_test = numeric_test[common_cols]
        
        # Reset indices to ensure alignment
        numeric_train = numeric_train.sort_index(axis=0).reset_index(drop=True)
        numeric_test = numeric_test.sort_index(axis=0).reset_index(drop=True)
        
        # Handle targets if provided
        if train_target is not None:
            train_target = train_target.sort_index(axis=0).reset_index(drop=True)
        if test_target is not None:
            test_target = test_target.sort_index(axis=0).reset_index(drop=True)
        
        # Verify shapes if targets are provided
        if train_target is not None and len(numeric_train) != len(train_target):
            raise ValueError(f"Train features {len(numeric_train)} and target {len(train_target)} length mismatch")
        if test_target is not None and len(numeric_test) != len(test_target):
            raise ValueError(f"Test features {len(numeric_test)} and target {len(test_target)} length mismatch")
        
        return numeric_train, numeric_test, train_target, test_target

    def _select_informative_features(self, train_features: pd.DataFrame, 
                                    test_features: pd.DataFrame) -> tuple:
        """Select most informative original features without transformations (unsupervised).
        Removes only features that are highly correlated (corr > 0.7) or have high mutual information (MI > 0.5) with embeddings."""
        # Calculate variance for all features
        variances = train_features.var()
        
        # Find all embedding features (features with 'emb' in name)
        emb_features = [col for col in train_features.columns if 'emb' in str(col).lower()]
        non_emb_features = [col for col in train_features.columns if 'emb' not in str(col).lower()]
        
        # Calculate correlation matrix
        corr_matrix = train_features.corr().abs()
        
        # Find features that are highly correlated with embeddings
        # Threshold for high correlation
        corr_threshold = 0.7
        features_to_remove_corr = set()
        
        # For each embedding feature, find all non-embedding features highly correlated with it
        for emb_feat in emb_features:
            if emb_feat in corr_matrix.columns:
                # Get correlations of all features with this embedding
                correlations = corr_matrix[emb_feat]
                # Find non-embedding features that are highly correlated with this embedding
                for feat in correlations.index:
                    if feat != emb_feat and 'emb' not in str(feat).lower():
                        if pd.notna(correlations[feat]) and correlations[feat] > corr_threshold:
                            features_to_remove_corr.add(feat)
        
        # Calculate mutual information for non-embedding features with embeddings
        mi_threshold = 0.5
        features_to_remove_mi = set()
        mi_scores_dict = {}
        
        for feat in non_emb_features:
            if feat not in train_features.columns:
                continue
            
            y_feat = train_features[feat].values
            
            # Skip if feature has no variance or is constant
            if np.var(y_feat) < 1e-10:
                continue
            
            # Calculate MI with each embedding and take maximum
            mi_scores = []
            for emb_feat in emb_features:
                try:
                    X_emb = train_features[[emb_feat]].values
                    mi = mutual_info_regression(
                        X_emb, 
                        y_feat, 
                        random_state=self.random_state
                    )[0]
                    mi_scores.append(mi)
                except Exception:
                    mi_scores.append(0)
            
            max_mi = np.max(mi_scores) if mi_scores else 0
            mi_scores_dict[feat] = max_mi
            
            # If mutual information is high, mark feature for removal
            if max_mi > mi_threshold:
                features_to_remove_mi.add(feat)
        
        # Combine features to remove (either high correlation OR high mutual information)
        features_to_remove = features_to_remove_corr.union(features_to_remove_mi)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': train_features.columns,
            'variance': variances
        })
        
        # Add correlation and MI metrics
        importance_df['max_correlation_with_emb'] = 0.0
        importance_df['max_mutual_info_with_emb'] = 0.0
        
        # Fill correlation values
        for feat in importance_df['feature']:
            if feat not in emb_features:
                max_corr = 0.0
                for emb_feat in emb_features:
                    if emb_feat in corr_matrix.columns and feat in corr_matrix.index:
                        corr_val = corr_matrix.loc[feat, emb_feat]
                        if pd.notna(corr_val):
                            max_corr = max(max_corr, abs(corr_val))
                importance_df.loc[importance_df['feature'] == feat, 'max_correlation_with_emb'] = max_corr
        
        # Fill MI values
        for feat, mi_val in mi_scores_dict.items():
            importance_df.loc[importance_df['feature'] == feat, 'max_mutual_info_with_emb'] = mi_val
        
        # Mark features to remove (highly correlated OR high mutual information with embeddings)
        importance_df['highly_correlated_with_emb'] = importance_df['feature'].isin(features_to_remove_corr)
        importance_df['high_mi_with_emb'] = importance_df['feature'].isin(features_to_remove_mi)
        importance_df['should_remove'] = importance_df['feature'].isin(features_to_remove)
        
        # Check if feature contains 'emb' in name
        importance_df['has_emb'] = importance_df['feature'].astype(str).str.lower().str.contains('emb', na=False)
        
        # Filter out features that are highly correlated OR have high mutual information with embeddings
        # Keep all embeddings and all other features that are not highly correlated/high MI with embeddings
        importance_df = importance_df[~importance_df['should_remove']]
        
        # Separate embeddings and non-embeddings
        emb_df = importance_df[importance_df['has_emb']].copy()
        non_emb_df = importance_df[~importance_df['has_emb']].copy()
        
        # Sort non-embeddings by variance (most informative first)
        non_emb_df = non_emb_df.sort_values(by='variance', ascending=False)
        
        # Select features: always keep all embeddings, then add non-embeddings
        if self.cols_budget is not None:
            # First, take all embeddings
            emb_cols = emb_df['feature'].tolist()
            n_emb = len(emb_cols)
            
            # Then, add non-embeddings up to budget
            remaining_budget = max(0, self.cols_budget - n_emb)
            non_emb_cols = non_emb_df['feature'].head(remaining_budget).tolist()
            
            top_cols = emb_cols + non_emb_cols
        else:
            # No budget: take all embeddings and all non-embeddings
            top_cols = emb_df['feature'].tolist() + non_emb_df['feature'].tolist()
        
        # Reconstruct importance_df with selected features in correct order
        # Create a mapping to preserve order
        order_map = {feat: idx for idx, feat in enumerate(top_cols)}
        selected_df = importance_df[importance_df['feature'].isin(top_cols)].copy()
        selected_df['_order'] = selected_df['feature'].map(order_map)
        selected_df = selected_df.sort_values('_order').drop('_order', axis=1)
        importance_df = selected_df
        
        # Return original unchanged features
        train_selected = train_features[top_cols]
        test_selected = test_features[top_cols]
        
        return train_selected, test_selected, top_cols, importance_df

    def _train_and_evaluate(self, train_features: pd.DataFrame, test_features: pd.DataFrame,
                           train_target: pd.Series, test_target: pd.Series) -> float:
        """Train model and evaluate on test set."""
        if self.model is None:
            raise ValueError("Model is required for downstream evaluation")
        
        model = clone(self.model)
        model.fit(train_features, train_target)
        
        if hasattr(model, 'predict_proba') and self.eval_metric == roc_auc_score:
            probs = model.predict_proba(test_features)[:, 1]
            score = self.eval_metric(test_target, probs)
        else:
            preds = model.predict(test_features)
            score = self.eval_metric(test_target, preds)
        
        return score

    def evaluate_feature_set(self, train_features: pd.DataFrame, test_features: pd.DataFrame, 
                            train_target: pd.Series = None, test_target: pd.Series = None) -> tuple:
        """Interpret feature set by selecting informative features without target supervision."""
        try:
            # Prepare and validate features
            numeric_train, numeric_test, train_target, test_target = self._prepare_features(
                train_features, test_features, train_target, test_target
            )
            
            # Select informative features without using target
            train_selected, test_selected, selected_cols, features_importance_df = self._select_informative_features(
                numeric_train, numeric_test
            )
            
            print(f"Selected {len(selected_cols)} informative features out of {len(numeric_train.columns)}")
            
            # Calculate downstream metric if enabled and targets are provided
            downstream_score = None
            if self.downstream_mode and train_target is not None and test_target is not None:
                if self.model is None:
                    warnings.warn("Model is required for downstream evaluation but not provided")
                else:
                    downstream_score = self._train_and_evaluate(
                        train_selected, test_selected, train_target, test_target
                    )
                    print(f"Downstream score: {downstream_score}")
            
            # Return selected columns with col_id
            selected_cols_with_id = [self.col_id] + selected_cols
            
            return downstream_score, selected_cols_with_id, features_importance_df
            
        except Exception as e:
            warnings.warn(f"Error interpreting feature set: {e}")
            return None, [], pd.DataFrame()
    
