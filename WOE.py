# Author:
# Sundar Krishnan
# Vlas Lezin  
# Sena Wright
# adopted from https://github.com/Sundar0989/WOE-and-IV/blob/master/WOE_IV.ipynb

"""
WEIGHT OF EVIDENCE (WOE) AND INFORMATION VALUE (IV) IMPLEMENTATION
==================================================================
This module implements WOE and IV calculations, which are fundamental
techniques in credit risk modeling. Through our coursework, we've learned
these methods are industry standards for feature engineering in financial
institutions.

Key concepts from our research:
- WOE transforms variables to show their predictive relationship with the target
- IV quantifies the overall predictive power of each variable
- Both metrics are essential for regulatory compliance in credit scoring

Mathematical foundation:
WOE = ln(% of Events / % of Non-Events)
IV = Σ (% Events - % Non-Events) × WOE

In our context:
- Events = Customers who made a purchase (target = 1)  
- Non-Events = Customers who did not purchase (target = 0)
"""

# import packages
import pandas as pd
import numpy as np
from pandas import Series
from scipy import stats
import re
import traceback
import string
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Global parameters control the binning process. Through trial and error,
we determined these defaults provide a good balance between granularity
and statistical significance.
"""
DEFAULT_MAX_BIN = 20
DEFAULT_FORCE_BIN = 3


def validate_inputs(func):
    """
    Input validation decorator to ensure data integrity before processing.
    
    During development, we encountered several edge cases that caused
    unexpected failures. This decorator addresses those systematically:
    - Ensures feature and target arrays have matching dimensions
    - Validates non-empty inputs
    - Confirms binary classification setup (0/1 target values only)
    
    Implementing this validation layer significantly improved our code's
    robustness and debugging efficiency.
    """
    def wrapper(Y, X, *args, **kwargs):
        if len(Y) != len(X):
            raise ValueError("Target (Y) and feature (X) must have the same length")
        if len(Y) == 0:
            raise ValueError("Input arrays cannot be empty")
        if not all(y in [0, 1] for y in Y):
            raise ValueError("Target variable must contain only 0 and 1 values")
        return func(Y, X, *args, **kwargs)
    return wrapper


@validate_inputs
def mono_bin(Y, X, n=None):

    if n is None:
        n = DEFAULT_MAX_BIN
    
    if n < 2:
        raise ValueError("Number of bins must be at least 2")
    
    # Separate missing and non-missing values for distinct treatment
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X', 'Y']][df1.X.isnull()]
    notmiss = df1[['X', 'Y']][df1.X.notnull()]
    
    # Ensure statistical significance with minimum observations per bin
    if len(notmiss) < n:
        logger.warning(f"Not enough non-missing values ({len(notmiss)}) for {n} bins. Adjusting bins.")
        n = max(2, len(notmiss) // 10)
    
    r = 0  # Spearman correlation coefficient
    d2 = None  # Grouped data
    
    # Iterative process to achieve monotonic binning
    while np.abs(r) < 1:
        try:
            # Use quantile-based binning for better handling of skewed distributions
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True, observed=False)
            
            if len(d2) < 2:
                n = n - 1
                if n < 2:
                    break
                continue
            
            # Spearman correlation assesses monotonic relationship
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1
            
        except (ValueError, KeyError) as e:
            logger.debug(f"Error in binning with n={n}: {e}")
            n = n - 1
            if n < 2:
                break
    
    # Fallback mechanism when monotonic binning isn't achievable
    if d2 is None or len(d2) == 1:
        n = DEFAULT_FORCE_BIN
        try:
            # Attempt quantile-based bins with forced count
            bins = notmiss.X.quantile(np.linspace(0, 1, n))
            if len(np.unique(bins)) == 2:
                bins = np.insert(bins, 0, notmiss.X.min() - 1)
                bins[1] = bins[1] - (bins[1] / 2)
        except Exception as e:
            logger.error(f"Failed to create bins: {e}")
            # Final fallback: equal-width bins
            bins = np.linspace(notmiss.X.min(), notmiss.X.max(), n + 1)
        
        d1 = pd.DataFrame({
            "X": notmiss.X, 
            "Y": notmiss.Y, 
            "Bucket": pd.cut(notmiss.X, np.unique(bins), include_lowest=True)
        })
        d2 = d1.groupby('Bucket', as_index=True, observed=False)
    
    # Calculate WOE and IV metrics
    d3 = _calculate_woe_iv(d2, justmiss)
    
    return d3


@validate_inputs
def char_bin(Y, X):

    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X', 'Y']][df1.X.isnull()]
    notmiss = df1[['X', 'Y']][df1.X.notnull()]
    
    if len(notmiss) == 0:
        logger.warning("All values are missing for this variable")
        return _create_empty_woe_df()
    
    # Group by unique categories
    df2 = notmiss.groupby('X', as_index=True)
    
    # Apply standard WOE/IV calculations
    d3 = _calculate_woe_iv(df2, justmiss)
    
    # For categorical variables, MIN and MAX values are identical
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    
    return d3


def _calculate_woe_iv(grouped_data, missing_data=None):
 
    d3 = pd.DataFrame()
    d3["COUNT"] = grouped_data.count().Y
    d3["EVENT"] = grouped_data.sum().Y

    try:
        # Extract bin boundaries for continuous variables
        d3["MIN_VALUE"] = grouped_data.min().X
        d3["MAX_VALUE"] = grouped_data.max().X
    except AttributeError:
        # For categorical variables, use the category itself
        d3["MIN_VALUE"] = d3.index
        d3["MAX_VALUE"] = d3.index
    
    d3["NONEVENT"] = d3["COUNT"] - d3["EVENT"]
    d3 = d3.reset_index(drop=True)
    
    # Incorporate missing values as a separate bin
    if missing_data is not None and len(missing_data.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE': np.nan}, index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = missing_data.count().Y
        d4["EVENT"] = missing_data.sum().Y
        d4["NONEVENT"] = d4["COUNT"] - d4["EVENT"]
        d3 = pd.concat([d3, d4], ignore_index=True)
    
    # Calculate totals for distribution computations
    total_events = d3['EVENT'].sum()
    total_nonevents = d3['NONEVENT'].sum()
    
    # Handle edge case where all observations belong to one class
    if total_events == 0 or total_nonevents == 0:
        logger.warning("No events or non-events in data. WOE/IV calculation may be unreliable.")
        d3["EVENT_RATE"] = 0
        d3["NON_EVENT_RATE"] = 0
        d3["DIST_EVENT"] = 0
        d3["DIST_NON_EVENT"] = 0
        d3["WOE"] = 0
        d3["IV"] = 0
    else:
        # Calculate event rates and distributions
        d3["EVENT_RATE"] = d3['EVENT'] / d3["COUNT"]
        d3["NON_EVENT_RATE"] = d3['NONEVENT'] / d3["COUNT"]
        
        # Apply epsilon to prevent numerical instabilities
        epsilon = 1e-10
        d3["DIST_EVENT"] = np.maximum(d3['EVENT'] / total_events, epsilon)
        d3["DIST_NON_EVENT"] = np.maximum(d3['NONEVENT'] / total_nonevents, epsilon)
        
        # Calculate Weight of Evidence
        d3["WOE"] = np.log(d3["DIST_EVENT"] / d3["DIST_NON_EVENT"])
        
        # Calculate Information Value contribution for each bin
        d3["IV"] = (d3["DIST_EVENT"] - d3["DIST_NON_EVENT"]) * d3["WOE"]
    
    d3["VAR_NAME"] = "VAR"
    
    # Structure output for clarity
    d3 = d3[['VAR_NAME', 'MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 
             'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT', 'DIST_NON_EVENT', 'WOE', 'IV']]
    
    # Replace infinities with zeros
    d3 = d3.replace([np.inf, -np.inf], 0)
    
    # Sum individual IVs for total Information Value
    d3["IV"] = d3["IV"].fillna(0)
    total_iv = d3["IV"].sum()
    d3.loc[:, "IV"] = total_iv  # Assign total to all rows for reference
    
    return d3

def _create_empty_woe_df():
    """
    Creates a properly structured empty DataFrame when no valid data exists.
    
    This ensures consistent output format across all scenarios, preventing
    downstream errors in the pipeline. The structure matches the expected
    output of successful WOE/IV calculations.
    """
    return pd.DataFrame({
        'VAR_NAME': ['VAR'],
        'MIN_VALUE': [np.nan],
        'MAX_VALUE': [np.nan],
        'COUNT': [0],
        'EVENT': [0],
        'EVENT_RATE': [0],
        'NONEVENT': [0],
        'NON_EVENT_RATE': [0],
        'DIST_EVENT': [0],
        'DIST_NON_EVENT': [0],
        'WOE': [0],
        'IV': [0]
    })


def data_vars(df1, target, max_bin=None, force_bin=None):
 
    # Update global parameters if provided
    global DEFAULT_MAX_BIN, DEFAULT_FORCE_BIN
    if max_bin is not None:
        DEFAULT_MAX_BIN = max_bin
    if force_bin is not None:
        DEFAULT_FORCE_BIN = force_bin
    
    # Validate inputs
    if len(df1) != len(target):
        raise ValueError("DataFrame and target must have the same length")
    
    # Extract target variable name from function call context
    # This prevents processing the target as a feature
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    
    try:
        vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
        final = (re.findall(r"[\w']+", vars_name))[-1]
    except:
        final = "TARGET"
        logger.warning("Could not extract target variable name, using 'TARGET'")
    
    # Process each variable
    x = df1.dtypes.index
    count = -1
    iv_df = None
    
    logger.info(f"Processing {len(x)} variables...")
    
    for i in x:
        # Skip the target variable
        if i.upper() == final.upper() or (hasattr(target, 'name') and i == target.name):
            continue
        
        try:
            # Apply appropriate binning based on variable type
            if pd.api.types.is_numeric_dtype(df1[i]) and df1[i].nunique() > 2:
                # Continuous variable: apply monotonic binning
                conv = mono_bin(target, df1[i])
            else:
                # Categorical variable: apply character binning
                conv = char_bin(target, df1[i])
            
            conv["VAR_NAME"] = i
            count = count + 1
            
            # Aggregate results
            if count == 0:
                iv_df = conv
            else:
                iv_df = pd.concat([iv_df, conv], ignore_index=True)
                
        except Exception as e:
            logger.error(f"Error processing variable {i}: {e}")
            continue  # Continue processing other variables
    
    # Handle case where no variables were successfully processed
    if iv_df is None:
        logger.warning("No variables were successfully processed")
        return pd.DataFrame(), pd.DataFrame()
    
    # Create summary table with total IV per variable
    iv = pd.DataFrame({'IV': iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    
    # Sort by IV value (highest predictive power first)
    iv = iv.sort_values('IV', ascending=False)
    
    # Add predictive power interpretation based on industry standards
    def interpret_iv(iv_value):
        if iv_value < 0.02:
            return "Not Predictive"
        elif iv_value < 0.1:
            return "Weak"
        elif iv_value < 0.3:
            return "Medium"
        elif iv_value < 0.5:
            return "Strong"
        else:
            return "Suspicious"
    
    iv['Predictive_Power'] = iv['IV'].apply(interpret_iv)
    
    logger.info(f"WOE/IV calculation completed for {len(iv)} variables")
    
    return (iv_df, iv)
