"""
Data Leakage Detection Utility

Detects common data leakage patterns in financial ML/RL training
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataLeakageDetector:
    """
    Detects data leakage in training/validation/test splits
    
    Common leakage sources:
    - Temporal leakage (train dates overlap with test)
    - Feature leakage (features use future data)
    - Target leakage (target calculated incorrectly)
    - Preprocessing leakage (normalization uses test data statistics)
    """
    
    def __init__(self):
        self.issues_found = []
        self.warnings = []
    
    def check_temporal_split(self, 
                            train_df: pd.DataFrame, 
                            val_df: pd.DataFrame, 
                            test_df: pd.DataFrame) -> bool:
        """
        Check for temporal data leakage in train/val/test split
        
        Returns True if no leakage detected
        """
        
        issues = []
        
        # Check 1: Temporal ordering
        train_max_date = train_df.index.max() if isinstance(train_df.index, pd.DatetimeIndex) else train_df.index[-1]
        val_min_date = val_df.index.min() if isinstance(val_df.index, pd.DatetimeIndex) else val_df.index[0]
        val_max_date = val_df.index.max() if isinstance(val_df.index, pd.DatetimeIndex) else val_df.index[-1]
        test_min_date = test_df.index.min() if isinstance(test_df.index, pd.DatetimeIndex) else test_df.index[0]
        
        if train_max_date >= val_min_date:
            issues.append(f"CRITICAL: Train data ({train_max_date}) overlaps with validation data ({val_min_date})")
        
        if val_max_date >= test_min_date:
            issues.append(f"CRITICAL: Validation data ({val_max_date}) overlaps with test data ({test_min_date})")
        
        # Check 2: No data gaps too large
        train_val_gap = (val_min_date - train_max_date).days if hasattr(train_max_date, 'days') else 1
        val_test_gap = (test_min_date - val_max_date).days if hasattr(val_max_date, 'days') else 1
        
        if train_val_gap > 30:
            self.warnings.append(f"WARNING: Large gap between train and validation: {train_val_gap} days")
        
        if val_test_gap > 30:
            self.warnings.append(f"WARNING: Large gap between validation and test: {val_test_gap} days")
        
        # Check 3: Sufficient data in each split
        min_train_size = 252  # At least 1 year
        min_val_size = 63    # At least 3 months
        min_test_size = 63   # At least 3 months
        
        if len(train_df) < min_train_size:
            self.warnings.append(f"WARNING: Training set small ({len(train_df)} < {min_train_size} recommended)")
        
        if len(val_df) < min_val_size:
            self.warnings.append(f"WARNING: Validation set small ({len(val_df)} < {min_val_size} recommended)")
        
        if len(test_df) < min_test_size:
            self.warnings.append(f"WARNING: Test set small ({len(test_df)} < {min_test_size} recommended)")
        
        self.issues_found.extend(issues)
        
        if issues:
            logger.error(f"Temporal leakage detected: {issues}")
            return False
        
        logger.info("✅ No temporal leakage detected")
        return True
    
    def check_feature_leakage(self, df: pd.DataFrame) -> bool:
        """
        Check for features that may contain future information
        
        Returns True if no leakage detected
        """
        
        issues = []
        
        # Check 1: Look for suspicious column names
        future_keywords = ['future', 'next', 'forward', 'ahead', 'tomorrow', 'target']
        
        for col in df.columns:
            col_lower = col.lower()
            for keyword in future_keywords:
                if keyword in col_lower and not col_lower.endswith('_lag'):
                    issues.append(f"SUSPICIOUS: Column '{col}' may contain future data")
        
        # Check 2: Check for NaN patterns that suggest backward filling
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isna().sum() > 0:
                # Check if NaN is at the start (expected) or end (suspicious)
                first_valid_idx = df[col].first_valid_index()
                last_valid_idx = df[col].last_valid_index()
                
                if first_valid_idx is not None and last_valid_idx is not None:
                    # NaN after last valid index is very suspicious
                    nan_after_last = df.loc[last_valid_idx:, col].isna().sum()
                    if nan_after_last > 1:
                        issues.append(f"SUSPICIOUS: Column '{col}' has NaN after last valid value (backward fill evidence?)")
        
        # Check 3: Check for inf values
        for col in df.select_dtypes(include=[np.number]).columns:
            if np.isinf(df[col]).any():
                issues.append(f"WARNING: Column '{col}' contains infinite values")
        
        self.issues_found.extend(issues)
        
        if issues:
            logger.error(f"Feature leakage risks detected: {issues}")
            return False
        
        logger.info("✅ No obvious feature leakage detected")
        return True
    
    def check_target_leakage(self, 
                            features_df: pd.DataFrame, 
                            target_col: str = None) -> bool:
        """
        Check if target variable leaks into features
        
        Returns True if no leakage detected
        """
        
        if target_col is None:
            return True
        
        issues = []
        
        # Check if target column exists in features
        if target_col in features_df.columns:
            issues.append(f"CRITICAL: Target column '{target_col}' found in features")
        
        # Check for high correlation with target
        if target_col in features_df.columns:
            target = features_df[target_col]
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col != target_col:
                    corr = features_df[col].corr(target)
                    if abs(corr) > 0.95:
                        issues.append(f"WARNING: Feature '{col}' highly correlated with target ({corr:.3f})")
        
        self.issues_found.extend(issues)
        
        if issues:
            logger.error(f"Target leakage detected: {issues}")
            return False
        
        return True
    
    def check_preprocessing_leakage(self, 
                                   train_stats: Dict, 
                                   val_stats: Dict, 
                                   test_stats: Dict) -> bool:
        """
        Check if preprocessing uses statistics from validation/test sets
        
        train_stats: Statistics calculated on training set
        val_stats/test_stats: Should be None or same as train_stats
        
        Returns True if no leakage detected
        """
        
        issues = []
        
        # If validation or test have different normalization stats, it's leakage
        if val_stats is not None and val_stats != train_stats:
            issues.append("WARNING: Validation set normalized with different statistics than training")
        
        if test_stats is not None and test_stats != train_stats:
            issues.append("WARNING: Test set normalized with different statistics than training")
        
        self.issues_found.extend(issues)
        
        if issues:
            logger.warning(f"Preprocessing leakage risks: {issues}")
            return False
        
        return True
    
    def run_full_check(self, 
                      train_df: pd.DataFrame,
                      val_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      target_col: str = None) -> Dict[str, Any]:
        """
        Run complete data leakage check
        
        Returns summary of all checks
        """
        
        logger.info("Running comprehensive data leakage detection...")
        
        self.issues_found = []
        self.warnings = []
        
        # Run all checks
        temporal_ok = self.check_temporal_split(train_df, val_df, test_df)
        feature_ok = self.check_feature_leakage(train_df)
        target_ok = self.check_target_leakage(train_df, target_col)
        
        # Summary
        all_passed = temporal_ok and feature_ok and target_ok
        
        summary = {
            'passed': all_passed,
            'temporal_check': temporal_ok,
            'feature_check': feature_ok,
            'target_check': target_ok,
            'critical_issues': [i for i in self.issues_found if 'CRITICAL' in i],
            'warnings': self.warnings + [i for i in self.issues_found if 'WARNING' in i or 'SUSPICIOUS' in i],
            'total_issues': len(self.issues_found),
            'total_warnings': len(self.warnings)
        }
        
        # Log results
        if all_passed:
            logger.info("✅ ALL DATA LEAKAGE CHECKS PASSED")
        else:
            logger.error(f"❌ DATA LEAKAGE DETECTED: {len(self.issues_found)} issues found")
            for issue in self.issues_found:
                logger.error(f"  - {issue}")
        
        if self.warnings:
            logger.warning(f"⚠️  {len(self.warnings)} warnings:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        return summary
    
    def validate_indicator_calculation(self, df: pd.DataFrame) -> bool:
        """
        Validate that indicators only use past data
        
        Checks for common mistakes in indicator calculation
        """
        
        issues = []
        
        # Check if any rolling calculations use future data
        # This is hard to detect automatically, but we can check window validity
        
        for col in df.columns:
            if 'SMA' in col or 'EMA' in col or 'MA' in col:
                # Moving averages should only have NaN at the start
                first_valid = df[col].first_valid_index()
                if first_valid is not None:
                    idx_pos = df.index.get_loc(first_valid)
                    # Should be at least window_size-1 rows before first valid
                    if idx_pos < 1:
                        self.warnings.append(f"SUSPICIOUS: {col} valid from row 0 (check window calculation)")
        
        return len(issues) == 0


def check_data_for_leakage(train_df: pd.DataFrame,
                          val_df: pd.DataFrame,
                          test_df: pd.DataFrame,
                          target_col: str = None) -> bool:
    """
    Convenience function to check for data leakage
    
    Returns True if no leakage detected
    """
    
    detector = DataLeakageDetector()
    summary = detector.run_full_check(train_df, val_df, test_df, target_col)
    
    if not summary['passed']:
        print("\n" + "="*60)
        print("❌ DATA LEAKAGE DETECTED - DO NOT USE THIS DATA FOR TRAINING")
        print("="*60)
        print(f"\nCritical Issues ({len(summary['critical_issues'])}):")
        for issue in summary['critical_issues']:
            print(f"  • {issue}")
        
        print(f"\nWarnings ({len(summary['warnings'])}):")
        for warning in summary['warnings'][:10]:  # Show first 10
            print(f"  • {warning}")
        
        print("\n" + "="*60)
        return False
    
    print("\n" + "="*60)
    print("✅ NO DATA LEAKAGE DETECTED - DATA IS SAFE FOR TRAINING")
    print("="*60)
    
    if summary['warnings']:
        print(f"\n⚠️  {len(summary['warnings'])} warnings to review:")
        for warning in summary['warnings'][:5]:
            print(f"  • {warning}")
    
    print("\n" + "="*60)
    return True


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    
    # Create sample data
    dates = pd.date_range('2022-01-01', periods=500, freq='D')
    
    # Simulate proper split
    train_df = pd.DataFrame({
        'Close': np.random.randn(350),
        'SMA_20': np.random.randn(350),
        'RSI': np.random.uniform(0, 100, 350)
    }, index=dates[:350])
    
    val_df = pd.DataFrame({
        'Close': np.random.randn(75),
        'SMA_20': np.random.randn(75),
        'RSI': np.random.uniform(0, 100, 75)
    }, index=dates[350:425])
    
    test_df = pd.DataFrame({
        'Close': np.random.randn(75),
        'SMA_20': np.random.randn(75),
        'RSI': np.random.uniform(0, 100, 75)
    }, index=dates[425:])
    
    # Run checks
    is_safe = check_data_for_leakage(train_df, val_df, test_df)
    print(f"\nData is {'SAFE' if is_safe else 'NOT SAFE'} for training")

