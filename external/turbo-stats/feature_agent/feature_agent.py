import logging
import pickle
from typing import List, Tuple, Optional, Union
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from code_execution import CodeExecutionEngine
from evaluation import SimpleModelEvaluator, SimpleModelInterpreter
from prompts import SYSTEM_PROMPT, PREVIOUS_FEATURES_TEMPLATE

class FeatureAgent:
    def __init__(
        self,
        llm,
        sequential_data: pd.DataFrame,
        transformed_data: pd.DataFrame,
        target: pd.Series,
        model,
        eval_metric=roc_auc_score,
        reflection: bool = True,
        iterations: int = 5,
        context_window: int = 5,
        temperature: float = 0.7,
        client_id_col: str = 'cl_id',
        test_size: float = 0.3,
        output_dir: str = None,
        random_state: int = 42,
        n_tries: int = 3,
        cols_budget: int = None,
        mode: str = 'interpretation'
    ):
        """Initialize feature generation agent using LLM for evolutionary feature search.
        
        Args:
            llm: Language model for generating feature code.
            sequential_data: Sequential data (transactions, events) with multiple records per client.
            transformed_data: Preprocessed data with existing features. Must contain client_id_col.
            target: Target variable aligned with transformed_data.
            model: ML model for feature evaluation (must support fit/predict).
            eval_metric: Evaluation metric function. Default: roc_auc_score.
            reflection: Whether to use reflection. Default: True.
            iterations: Number of evolutionary search iterations. Default: 5.
            context_window: Number of best features from memory included in LLM context. Default: 5.
            temperature: LLM temperature for generation. Default: 0.7.
            client_id_col: Client ID column name. Default: 'cl_id'.
            test_size: Test set proportion. Default: 0.3.
            output_dir: Directory to save the model. Default: None.
            random_state: Random seed for reproducibility. Default: 42.
            n_tries: Number of generation attempts per iteration. Default: 3.
            cols_budget: Number of columns budget. Default: None.
            mode: Mode of operation. Either 'performance' or 'interpretation'. Default: 'performance'.
        Raises:
            ValueError: If client_id_col missing, data length mismatch, data leakage detected, or invalid mode.
        """
        # Validate mode
        if mode not in ['performance', 'interpretation']:
            raise ValueError(f"Mode must be either 'performance' or 'interpretation', got '{mode}'")
        
        self.llm = llm
        self.sequential_data = sequential_data
        self.transformed_data = transformed_data
        self.target = target
        self.model = model
        self.reflection = reflection
        self.iterations = iterations
        self.memory = []
        self.context_window = context_window
        self.cols_budget = cols_budget
        self.best_score = -np.inf
        self.client_id_col = client_id_col
        self.test_size = test_size
        self.random_state = random_state
        self.n_tries = n_tries
        self.output_dir = output_dir
        self.mode = mode

        self.generated_features = []
        self.scores = []
        # Internal train-test split
        self.sequential_train, self.sequential_test = None, None
        self.transformed_train, self.transformed_test = None, None
        self.target_train, self.target_test = None, None
        
        # Set up logging
        self.logger = logging.getLogger("FeatureAgent")
        
        # Initialize components
        if self.mode == 'performance':
            self.evaluator = SimpleModelEvaluator(model, eval_metric, random_state, client_id_col, target.name, cols_budget)
        else:
            self.evaluator = SimpleModelInterpreter(model, eval_metric, random_state, client_id_col, target.name, cols_budget)
        self.code_engine = CodeExecutionEngine(client_id_col)

        self.previous_error = None
        self.error_code = None
        self.previous_bad_features = []
        self.current_features_importances_df = None  # Store current feature importances
        
        # Validate and split data
        self._validate_and_split_data()

    def _extract_code_from_response(self, response_text: str) -> str:
        """Extract Python code from the LLM response using code engine."""
        return self.code_engine.extract_code_from_response(response_text)

    def _execute_sequential_feature_code(self, 
                                        code: str, 
                                        train_sequential_df: pd.DataFrame, 
                                        test_sequential_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute sequential feature code using code engine."""
        return self.code_engine.execute_sequential_feature_code(code, train_sequential_df, test_sequential_df)

    def _combine_features(self, transformed_df: pd.DataFrame, new_features_df: pd.DataFrame) -> pd.DataFrame:
        """Combine new sequential features with existing transformed features."""
        # Merge new features with transformed data
        combined_data = transformed_df.merge(
            new_features_df, 
            on=self.client_id_col, 
            how='left'
        )
        
        # Fill NaN values with appropriate defaults
        numeric_cols = combined_data.select_dtypes(include=[np.number]).columns
        combined_data[numeric_cols] = combined_data[numeric_cols].fillna(0)
        
        self.logger.info(f"Combined features: {len(transformed_df.columns)} original + {len(new_features_df.columns) - 1} new = {len(combined_data.columns)} total")
        
        return combined_data

    def _validate_and_split_data(self):
        """Validate data and perform proper train-test split without leakage."""
        if self.client_id_col not in self.sequential_data.columns:
            raise ValueError(f"Client ID column '{self.client_id_col}' not found in sequential data")
        if self.client_id_col not in self.transformed_data.columns:
            raise ValueError(f"Client ID column '{self.client_id_col}' not found in transformed data")
        
        # Ensure target is properly aligned with transformed_data
        if len(self.target) != len(self.transformed_data):
            raise ValueError(f"Target length {len(self.target)} doesn't match transformed data length {len(self.transformed_data)}")
        
        # Get unique clients from transformed data (which has the target)
        clients = self.transformed_data[self.client_id_col].unique()
        
        # Use the target for stratification if binary/multi-class
        unique_targets = self.target.nunique()
        if unique_targets > 1 and unique_targets <= 10:
            stratify = self.target
        else:  # Regression or too many classes
            stratify = None
        
        train_clients, test_clients = train_test_split(
            clients, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=stratify
        )
        
        # Split sequential data
        self.sequential_train = self.sequential_data[
            self.sequential_data[self.client_id_col].isin(train_clients)
        ].copy()
        self.sequential_test = self.sequential_data[
            self.sequential_data[self.client_id_col].isin(test_clients)
        ].copy()
        
        # Split transformed data
        self.transformed_train = self.transformed_data[
            self.transformed_data[self.client_id_col].isin(train_clients)
        ].copy()
        self.transformed_test = self.transformed_data[
            self.transformed_data[self.client_id_col].isin(test_clients)
        ].copy()
        
        # Split target by the same client split
        train_mask = self.transformed_data[self.client_id_col].isin(train_clients)
        test_mask = self.transformed_data[self.client_id_col].isin(test_clients)
        
        self.target_train = self.target[train_mask].sort_index(axis=0).reset_index(drop=True)
        self.target_test = self.target[test_mask].sort_index(axis=0).reset_index(drop=True)
        
        # Reset indices to avoid alignment issues
        self.transformed_train = self.transformed_train.sort_index(axis=0).reset_index(drop=True)
        self.transformed_test = self.transformed_test.sort_index(axis=0).reset_index(drop=True)
        
        # Verify no data leakage
        if set(train_clients) & set(test_clients):
            raise ValueError("Data leakage detected: train and test clients overlap!")
        self.system_prompt = self._build_system_prompt()
        self.logger.info(f"Train clients: {len(train_clients)}, Test clients: {len(test_clients)}")
        self.logger.info(f"Sequential train: {len(self.sequential_train)} rows, {self.sequential_train[self.client_id_col].nunique()} clients")
        self.logger.info(f"Sequential test: {len(self.sequential_test)} rows, {self.sequential_test[self.client_id_col].nunique()} clients")
        self.logger.info(f"Target train distribution: {pd.Series(self.target_train).value_counts().to_dict()}")
        self.logger.info(f"Target test distribution: {pd.Series(self.target_test).value_counts().to_dict()}")

    def _evaluate_feature_set(
        self,
        train_features: pd.DataFrame,
        test_features: pd.DataFrame,
        train_target: pd.Series,
        test_target: pd.Series
    ) -> float:
        """Evaluate feature set using the centralized evaluator."""
        # self.evaluator.debug_evaluation(
        #     train_features, test_features, train_target, test_target
        # )
        return self.evaluator.evaluate_feature_set(
            train_features, test_features, train_target, test_target
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt for the LLM."""
        categorical_columns = [col for col in self.sequential_train.columns if self.sequential_train[col].dtype == 'int64']
        categorical_unique_counts = {col: self.sequential_train[col].nunique() for col in categorical_columns}
        target_unique = self.target_train.nunique()
        if target_unique == 2:
            target_type = 'Binary'
        elif target_unique > 2:
            target_type = 'Multi-class'
        else:
            target_type = 'Regression'
        return SYSTEM_PROMPT.format(
            target_name=self.target.name,
            sequential_train_count=len(self.sequential_train),
            sequential_train_clients=self.sequential_train[self.client_id_col].nunique(),
            sequential_columns=', '.join(self.sequential_train.columns.tolist()),
            sequential_categorical_columns=', '.join(categorical_columns),
            sequential_dtypes=', '.join([str(dtype) for dtype in self.sequential_train.dtypes.tolist()]),
            categorical_unique_counts=', '.join([f"{col}: {count}" for col, count in categorical_unique_counts.items()]),
            sequential_head=self.sequential_train.head().to_string(),
            target_distribution=', '.join([f"{key}: {value}" for key, value in pd.Series(self.target_train).value_counts().to_dict().items()]),
            target_type=target_type,
        )
    def _build_previous_features(self) -> str:
        """Build previous features section for the LLM."""        
        previous_features_section = ""
        if self.memory:
            previous_features_section = PREVIOUS_FEATURES_TEMPLATE.format(
                context_window=self.context_window,
                previous_features="\n".join([
                    f"  Feature name: {code}... (score: {score:.4f})"
                    for (code, score, _) in sorted(self.memory, key=lambda x: x[1], reverse=True) 
                ]),
                previous_bad_features="\n".join([
                    f"  Feature name: {code}"
                    for code in self.previous_bad_features
                ])
            )
        
        return previous_features_section

    def _generate_feature_code(self) -> str:
        """Generate feature code with validation and multiple attempts."""
        pass

    def _attempt_feature_generation(
        self, iteration: int
    ) -> Tuple[Optional[str], float, Optional[Tuple[pd.DataFrame, pd.DataFrame]]]:
        """Attempt to generate and evaluate a new feature with multiple tries.
        
        Returns:
            Tuple of (best_code, best_score, best_features) where best_features
            is a tuple of (combined_train, combined_test) DataFrames.
        """
        best_attempt_score = -np.inf
        best_attempt_code = None
        best_attempt_features = None
        success = False
        attempt = 0
        new_cols = self.transformed_train.columns.tolist()
        features_importances_df = pd.DataFrame(
        )
        while not success:
            if attempt > self.n_tries:
                break
            attempt += 1
            self.logger.info(f"  Attempt {attempt} for iteration {iteration + 1}")
            
            # Generate new feature code for sequential data
            new_feature_code = self._generate_feature_code()
            
            if not new_feature_code:
                self.logger.warning("Failed to generate feature code, skipping attempt")
                continue
                
            try:
                # Execute the code to create new features from train sequential data
                self.logger.debug(f"Generated feature code:\n{new_feature_code}")
                new_features_train, new_features_test = self._execute_sequential_feature_code(
                    new_feature_code, 
                    self.sequential_train, 
                    self.sequential_test
                )
                
                self.generated_features.append([new_features_train.copy(), new_features_test.copy()])
                # Only rename columns that are not self.client_id_col
                self.generated_features[-1][0].columns = [
                    f"iter_{iteration + 1}_{col}" if col != self.client_id_col else col
                    for col in self.generated_features[-1][0].columns
                ]
                self.generated_features[-1][1].columns = [
                    f"iter_{iteration + 1}_{col}" if col != self.client_id_col else col
                    for col in self.generated_features[-1][1].columns
                ]
                # Combine with existing transformed features
                combined_train = self._combine_features(
                    self.transformed_train, new_features_train
                )
                combined_test = self._combine_features(
                    self.transformed_test, new_features_test
                )
                
                # Evaluate the combined feature set using centralized evaluator
                new_score, new_cols, features_importances_df = self._evaluate_feature_set(
                    combined_train, combined_test, self.target_train, self.target_test
                )
                
                self.logger.info(f"    Attempt score: {new_score:.4f}")
                
                # Track the best attempt
                if new_score > best_attempt_score:
                    best_attempt_score = new_score
                    best_attempt_code = new_feature_code
                    combined_train, combined_test = combined_train[new_cols], combined_test[new_cols]
                    best_attempt_features = (combined_train, combined_test)

                success = True
                self.previous_error = None
                self.error_code = None
            except Exception as e:
                self.logger.warning(f"    Error in attempt {attempt}: {e}")
                self.previous_error = e
                self.error_code = new_feature_code
                continue
        
        return best_attempt_code, best_attempt_score, best_attempt_features, new_cols, features_importances_df

    def evolutionary_search(self) -> List[Tuple[str, float]]:
        """Run the evolutionary feature generation process with multiple attempts per iteration."""
        self.logger.info("Starting evolutionary sequential feature search...")
        
        # Evaluate baseline (only pre-transformed features)
        baseline_score, baseline_cols, baseline_features_importances_df = self._evaluate_feature_set(
            self.transformed_train, self.transformed_test,
            self.target_train, self.target_test
        )
        # Store baseline feature importances for use in first iteration's reflection
        if baseline_features_importances_df is not None and not baseline_features_importances_df.empty:
            self.current_features_importances_df = baseline_features_importances_df
        self.best_score = baseline_score
        self.scores.append((0, baseline_score))
        self.logger.info(f"Baseline score (pre-transformed features only): {baseline_score:.4f}")
        
        for iteration in range(self.iterations):
            self.logger.info(f"Iteration {iteration + 1}/{self.iterations}")
            self.previous_error = None
            self.error_code = None
            # Attempt feature generation with multiple tries
            code, score, features, new_cols, features_importances_df = self._attempt_feature_generation(iteration)
            
            # Store current feature importances for use in next iteration's reflection
            if features_importances_df is not None and not features_importances_df.empty:
                self.current_features_importances_df = features_importances_df
            
            if code and score > 0:
                self.logger.info(f"Attempt score: {score:.4f} (vs best attempt: {self.best_score:.4f} vs baseline: {baseline_score:.4f})")
                self.scores.append((iteration+1, score))
                
                # In interpretation mode, always update memory regardless of score
                # In performance mode, only update if score improves
                if self.mode == 'interpretation':
                    self.memory.append((code, score, new_cols))
                    self.transformed_train, self.transformed_test = features
                    if score > self.best_score:
                        self.best_score = score
                    self.logger.info("✅ Feature added to memory (interpretation mode).")
                else:  # performance mode
                    # Update memory if improvement
                    if score > self.best_score:
                        self.memory.append((code, score, new_cols))
                        self.best_score = score
                        self.transformed_train, self.transformed_test = features
                        self.logger.info("🎯 New best score! Added to memory.")
                    else:
                        self.logger.info("No improvement over current best, feature discarded")
                        self.previous_bad_features.append(code)
            else:
                self.logger.warning("No valid features generated in this iteration")
                
        self.logger.info(f"Evolutionary search completed. Best score: {self.best_score:.4f}")
        self.logger.info(f"Generated {len(self.memory)} successful features")
        if self.output_dir:
            self.save(self.output_dir, save_scores=True, save_plot=True, save_generated_features=True)
        return self.memory

    def get_best_features(self) -> List[Tuple[str, float]]: 
        """Get the best features from memory sorted by score."""
        return self.memory

    def apply_best_features_to_full_data(self, n_features: Optional[int] = None, 
                                   sequential_data_train: Optional[pd.DataFrame] = None,
                                   sequential_data_test: Optional[pd.DataFrame] = None,
                                   transformed_data_train: Optional[pd.DataFrame] = None,
                                   transformed_data_test: Optional[pd.DataFrame] = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """Apply features from memory to any data.
        
        Args:
            n_features: Number of top features to apply. If None, applies all features.
            external_sequential_data: External sequential data to apply features to. 
                                    If None, uses internal training data.
            external_transformed_data: External transformed data to merge features with.
                                     If None, uses internal transformed data.
        
        Returns:
            Enhanced dataset(s)
        """
        best_features = self.get_best_features()  # All features
        if n_features is not None:
            best_features = best_features[:n_features]
        if sequential_data_train is not None and sequential_data_test is not None and transformed_data_train is not None and transformed_data_test is not None:
            # Apply to external data - return single enhanced dataset
            result_train = transformed_data_train.copy()
            result_test = transformed_data_test.copy()
            
            for i, (code, score, cols) in enumerate(best_features):
                try:
                    # Execute feature code on external sequential data
                    new_features_train, new_features_test = self._execute_sequential_feature_code(code, sequential_data_train, sequential_data_test)
                    
                    # Merge with external transformed data
                    result_train = result_train.merge(
                        new_features_train, 
                        on=self.client_id_col, 
                        how='left'
                    )
                    result_test = result_test.merge(
                        new_features_test,
                        on=self.client_id_col,
                        how='left'
                    )
                    intersection_cols = list(set(cols) & set(result_train.columns) & set(result_test.columns))
                    result_train, result_test = result_train[intersection_cols], result_test[intersection_cols]
                    feature_count = len(new_features_train.columns) - 1
                    self.logger.info(f"Applied {feature_count} features from feature_{i+1} (score: {score:.4f})")
                except Exception as e:
                    self.logger.warning(f"Failed to apply feature {i+1}: {e}")
            
            # Fill NaN values
            numeric_cols_train = result_train.select_dtypes(include=[np.number]).columns
            numeric_cols_test = result_test.select_dtypes(include=[np.number]).columns
            result_train[numeric_cols_train] = result_train[numeric_cols_train].fillna(0)
            result_test[numeric_cols_test] = result_test[numeric_cols_test].fillna(0)
            
            self.logger.info(f"Final enhanced dataset: {len(result_train.columns)} total features")
            return result_train, result_test
            
        else:
            raise ValueError("External data is not provided")

    
    def get_feature_count(self) -> int:
        """Get the total number of successful features in memory."""
        return len(self.memory)
    
    def get_feature_scores(self) -> List[Tuple[str, float]]:
        """Get all features with their scores sorted by performance."""
        return self.get_best_features()

    # def get_feature_importance(self, n_features: int = 5) -> pd.DataFrame:
    #     """Get feature importance for the best feature set."""
    #     try:
    #         # Apply best features to get enhanced dataset
    #         train_enhanced, test_enhanced = self.apply_best_features_to_full_data(n_features)
            
    #         # Prepare data for importance calculation
    #         X_train = train_enhanced.drop(columns=[self.client_id_col], errors='ignore')
    #         X_train = X_train.select_dtypes(include=[np.number])
    #         y_train = self.target
            
    #         # Train model on full enhanced data
    #         self.model.fit(X_train, y_train)
            
    #         # Get feature importance
    #         if hasattr(self.model, 'get_feature_importance'):
    #             importance = self.model.get_feature_importance()
    #         elif hasattr(self.model, 'feature_importances_'):
    #             importance = self.model.feature_importances_
    #         else:
    #             self.logger.warning("Model doesn't support feature importance")
    #             return pd.DataFrame()
            
    #         # Create importance DataFrame
    #         importance_df = pd.DataFrame({
    #             'feature': X_train.columns,
    #             'importance': importance
    #         }).sort_values('importance', ascending=False)
            
    #         return importance_df
            
    #     except Exception as e:
    #         self.logger.error(f"Error calculating feature importance: {e}")
    #         return pd.DataFrame()

    def save(self, output_dir: str, save_scores: bool = False, save_plot: bool = False, save_generated_features: bool = False):
        """Save the model to a file."""
        os.makedirs(output_dir, exist_ok=True)
        if save_scores:
            scores_csv_path = os.path.join(output_dir, 'feature_agent_scores.csv')
        scores_df = pd.DataFrame(self.scores, columns=['iteration', 'score'])
        scores_df.to_csv(scores_csv_path, index=False)
        if save_generated_features:
            train_features = self.transformed_train.copy()
            test_features = self.transformed_test.copy()
            for i, (train, test) in enumerate(self.generated_features):
                train_features = train_features.merge(train, on=self.client_id_col, how='left')
                test_features = test_features.merge(test, on=self.client_id_col, how='left')
            train_features.to_csv(os.path.join(output_dir, 'generated_features_train.csv'), index=False)
            test_features.to_csv(os.path.join(output_dir, 'generated_features_test.csv'), index=False)
                
        if save_plot:
            plot_path = os.path.join(output_dir, 'feature_agent_scores.png')
            plt.figure(figsize=(8, 5))
            plt.plot(scores_df['iteration'], scores_df['score'], marker='o', linestyle='-', color='b', label='Score')
            plt.xlabel('Iteration')
            plt.ylabel('Score')
            plt.title('Feature Agent Evolutionary Search Scores')
            plt.legend()
            plt.grid(True)
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
        dict_to_save = {
            'memory': self.memory,
        }
        with open(os.path.join(output_dir, 'feature_agent.pkl'), 'wb') as f:
            pickle.dump(dict_to_save, f)
    
    def load(self, output_dir: str):
        """Load the model from a file."""
        with open(os.path.join(output_dir, 'feature_agent.pkl'), 'rb') as f:
            dict_to_load = pickle.load(f)
        self.memory = dict_to_load['memory']