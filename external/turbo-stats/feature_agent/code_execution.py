import pandas as pd
import numpy as np
import logging
import textwrap
import traceback
from typing import Optional, Tuple, List
from code_security_scanner import CodeSecurityScanner
import sys

class CodeExecutionEngine:
    """Handles safe extraction and execution of LLM-generated feature code."""
    
    def __init__(self, client_id_col: str = 'cl_id'):
        self.client_id_col = client_id_col
        self.code_security_scanner = CodeSecurityScanner()
        self.logger = logging.getLogger("CodeExecutionEngine")
    
    def extract_code_from_response(self, response_text: str) -> Optional[str]:
        """Extract Python code from the LLM response."""
        try:
            # If the response contains code blocks, extract them
            if "```python" in response_text:
                code_start = response_text.find("```python") + 9
                code_end = response_text.find("```", code_start)
                code = response_text[code_start:code_end]
            elif "```" in response_text:
                code_start = response_text.find("```") + 3
                code_end = response_text.find("```", code_start)
                code = response_text[code_start:code_end].strip()
            else:
                # Assume the entire response is code
                code = response_text.strip()
                
            return code
            
        except Exception:
            error_str = traceback.format_exc()
            self.logger.error(f"Error extracting code from response: {error_str}")
            return None

    def execute_sequential_feature_code(self, 
                                        code: str, 
                                        train_sequential_df: pd.DataFrame, 
                                        test_sequential_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Safely execute the generated sequential feature code with enhanced validation."""
        try:
            # Add required imports to the code
            indented_code = textwrap.dedent(code)
            full_code = self._build_safe_execution_code(indented_code)

            # Create a safe execution environment
            safe_globals = self._create_safe_environment(train_sequential_df, test_sequential_df)
            
            # Scan the code for security vulnerabilities
            vulnerabilities = self.code_security_scanner.scan_comprehensive(full_code)
            vulnerabilities_report = self.code_security_scanner.format_report(vulnerabilities)
            if vulnerabilities_report:
                raise ValueError(f"Security vulnerabilities detected in the code: {vulnerabilities_report}")
            
            # Execute the code to define the function
            exec(full_code, safe_globals)
            
            # Find and execute the feature creation function
            function_name = self._find_feature_function(safe_globals)
            feature_func = safe_globals[function_name]
            new_train_features_df, new_test_features_df = feature_func(train_sequential_df, test_sequential_df, self.client_id_col)
            
            # Validate and clean the resulting DataFrame
            new_train_features_df = self._validate_and_clean_features(new_train_features_df)
            new_test_features_df = self._validate_and_clean_features(new_test_features_df)
            
            self.logger.info(f"Successfully executed feature code. Generated {len(new_train_features_df.columns) - 1} features")
            return new_train_features_df, new_test_features_df
            
        except Exception as e:
            # 1. Получаем информацию об исключении
            exc_type, exc_value, exc_traceback = sys.exc_info()
            
            # 2. Извлекаем стек вызовов как список объектов
            tb_list = traceback.extract_tb(exc_traceback)
            
            # 3. Ищем номер строки именно внутри нашего exec-кода (файл "<string>")
            error_line_number = None
            problematic_line_content = "Could not extract specific line."
            
            # Идем с конца стека, чтобы найти самую глубокую точку ошибки в нашем коде
            for frame in reversed(tb_list):
                if frame.filename == "<string>":
                    error_line_number = frame.lineno
                    break
            
            # 4. Формируем полный код с нумерацией для контекста
            full_code_lines = full_code.split('\n') if full_code else []
            numbered_code = "\n".join([f"{i+1:>4}: {line}" for i, line in enumerate(full_code_lines)])
            
            # 5. Вытаскиваем конкретную строку ошибки, если нашли номер
            if error_line_number is not None and 0 <= error_line_number - 1 < len(full_code_lines):
                raw_line = full_code_lines[error_line_number - 1].strip()
                problematic_line_content = f"Line {error_line_number}: {raw_line}"
            
            # 6. Собираем полное сообщение об ошибке
            error_str = traceback.format_exc()
            detailed_msg = (
                f"Error executing feature code: {str(e)}\n"
                f"🔴 PROBLEM: {problematic_line_content}\n\n" 
                f"--- EXECUTED CODE CONTEXT ---\n"
                f"{numbered_code}\n"
                f"-----------------------------\n"
                f"Full traceback:\n{error_str}"
            )
            
            # self.logger.error(detailed_msg)
            raise type(e)(detailed_msg) from e

    def _build_safe_execution_code(self, indented_code: str) -> str:
        """Build the safe execution code with required imports and utilities."""
        return f"""
import pandas as pd
import numpy as np
import re
from datetime import datetime

# Safe functions from FeatureFactoryRosbank
def _safe_div(a, b):
    return np.where(b != 0, a / b, 0.0)

def entropy_from_counts(cnts: np.ndarray) -> float:
    total = cnts.sum()
    if total <= 0:
        return 0.0
    p = cnts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def hhi_from_counts(cnts: np.ndarray) -> float:
    total = cnts.sum()
    if total <= 0:
        return 0.0
    p = cnts / total
    return float((p * p).sum())

def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or np.all(x == 0):
        return 0.0
    x = np.sort(np.abs(x))
    n = x.size
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * (cumx.sum() / cumx[-1])) / n)

{indented_code}
"""

    def _create_safe_environment(self, train_sequential_df: pd.DataFrame, test_sequential_df: pd.DataFrame) -> dict:
        """Create a safe execution environment with restricted access."""
        return {
                'pd': pd,
                'np': np,
                'train_seq_df': train_sequential_df.copy(),  # Use copy to avoid modifying original
                'test_seq_df': test_sequential_df.copy(),  # Use copy to avoid modifying original
                'client_id_col': self.client_id_col,
                '__builtins__': __builtins__
            }

    def _find_feature_function(self, safe_globals: dict) -> str:
        """Find the feature creation function in the executed globals."""
        function_name = None
        for key in safe_globals:
            if key.startswith('create_feature') and callable(safe_globals[key]):
                function_name = key
                break
        
        if not function_name:
            raise ValueError("No feature creation function found in the code")
        
        return function_name

    def _validate_and_clean_features(self, new_features_df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the generated features DataFrame."""
        # Enhanced validation
        if not isinstance(new_features_df, pd.DataFrame):
            raise ValueError(f"Function returned {type(new_features_df)}, expected DataFrame")
        
        if len(new_features_df) == 0:
            raise ValueError("Function returned empty DataFrame")
        
        # Check for client_id_col in index or columns
        if (self.client_id_col not in new_features_df.index and 
            self.client_id_col not in new_features_df.columns):
            raise ValueError(f"Result must contain '{self.client_id_col}'")
        
        # Ensure client_id is a column
        if self.client_id_col in new_features_df.index:
            new_features_df = new_features_df.reset_index()
        
        # Validate data types and clean up
        for col in new_features_df.columns:
            if col != self.client_id_col:
                # Convert to numeric, coercing errors
                new_features_df[col] = pd.to_numeric(new_features_df[col], errors='coerce')
        
        # Final cleanup
        new_features_df = new_features_df.fillna(0)
        new_features_df = new_features_df.replace([np.inf, -np.inf], 0.0)
        
        # Remove duplicate client_ids
        new_features_df = new_features_df.drop_duplicates(subset=[self.client_id_col])
        
        return new_features_df