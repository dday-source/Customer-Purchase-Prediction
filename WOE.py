# Author:
# Sundar Krishnan
# Vlas Lezin  
# Sena Wright
# adopted from https://github.com/Sundar0989/WOE-and-IV/blob/master/WOE_IV.ipynb

"""
=================================================================================
EDUCATIONAL ANNOTATIONS FOR WEIGHT OF EVIDENCE (WOE) AND INFORMATION VALUE (IV)
=================================================================================
This module implements WOE and IV calculations, which are fundamental techniques
in financial risk modeling and credit scoring.

WHAT WOE DOES:
- Transforms variables to show their relationship with the target
- Handles both continuous and categorical variables
- Creates monotonic relationships for better model interpretability

WHAT IV DOES:
- Measures the predictive power of each variable
- Helps select the most important features
- Standard metric in financial modeling

KEY CONCEPTS:
WOE = ln(% of Events / % of Non-Events)
IV = Σ (% Events - % Non-Events) × WOE

WHERE:
- Events = Customers who purchased (target = 1)
- Non-Events = Customers who didn't purchase (target = 0)
=================================================================================
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
GLOBAL PARAMETERS:
These control the binning behavior for continuous variables
- max_bin: Start with 20 bins and reduce until monotonic relationship found
- force_bin: If monotonic binning fails, force this many bins
"""
DEFAULT_MAX_BIN = 20
DEFAULT_FORCE_BIN = 3


def validate_inputs(func):
    """
    WHAT: Decorator that validates inputs before processing
    
    HOW IT WORKS:
    1. Checks that feature and target have same length
    2. Ensures arrays aren't empty
    3. Verifies target is binary (only 0s and 1s)
    
    WHY IMPORTANT:
    - Prevents crashes from bad data
    - Provides clear error messages
    - Ensures WOE/IV calculations are valid
    
    DECORATOR PATTERN:
    This wraps other functions to add validation without modifying them
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
    """
    WHAT: Creates bins for continuous variables while maintaining monotonic relationship
    
    MONOTONIC RELATIONSHIP EXPLAINED:
    - As the feature value increases, the event rate should consistently increase OR decrease
    - Example: Higher income → Higher purchase probability (monotonic increasing)
    - This makes the model more interpretable and stable
    
    HOW THE ALGORITHM WORKS:
    1. Start with many bins (default 20)
    2. Calculate Spearman correlation between bin averages and event rates
    3. If correlation isn't perfect (±1), reduce bins and try again
    4. Continue until monotonic relationship achieved
    5. If it fails, force a small number of bins
    
    PARAMETERS:
    Y : Binary target (0 or 1)
    X : Continuous feature to bin
    n : Maximum number of bins to try
    
    RETURNS:
    DataFrame with WOE and IV for each bin
    """
    if n is None:
        n = DEFAULT_MAX_BIN
    
    if n < 2:
        raise ValueError("Number of bins must be at least 2")
    
    # Separate missing and non-missing values
    # Missing values get their own "bin" for WOE calculation
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X', 'Y']][df1.X.isnull()]
    notmiss = df1[['X', 'Y']][df1.X.notnull()]
    
    # Ensure enough data points per bin (at least 10)
    if len(notmiss) < n:
        logger.warning(f"Not enough non-missing values ({len(notmiss)}) for {n} bins. Adjusting bins.")
        n = max(2, len(notmiss) // 10)
    
    r = 0  # Correlation coefficient
    d2 = None  # Grouped data
    
    # MONOTONIC BINNING LOOP
    # Keep reducing bins until we get a perfect monotonic relationship
    while np.abs(r) < 1:
        try:
            # Create quantile-based bins (equal number of observations per bin)
            # This is better than equal-width bins for skewed distributions
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True, observed=False)
            
            if len(d2) < 2:
                n = n - 1
                if n < 2:
                    break
                continue
            
            # Spearman correlation checks monotonic relationship
            # r = 1: perfect increasing, r = -1: perfect decreasing
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1
            
        except (ValueError, KeyError) as e:
            logger.debug(f"Error in binning with n={n}: {e}")
            n = n - 1
            if n < 2:
                break
    
    # FALLBACK MECHANISM
    # If monotonic binning fails, force a small number of bins
    if d2 is None or len(d2) == 1:
        n = DEFAULT_FORCE_BIN
        try:
            # Try quantile-based bins first
            bins = notmiss.X.quantile(np.linspace(0, 1, n))
            if len(np.unique(bins)) == 2:
                bins = np.insert(bins, 0, notmiss.X.min() - 1)
                bins[1] = bins[1] - (bins[1] / 2)
        except Exception as e:
            logger.error(f"Failed to create bins: {e}")
            # Last resort: equal-width bins
            bins = np.linspace(notmiss.X.min(), notmiss.X.max(), n + 1)
        
        d1 = pd.DataFrame({
            "X": notmiss.X, 
            "Y": notmiss.Y, 
            "Bucket": pd.cut(notmiss.X, np.unique(bins), include_lowest=True)
        })
        d2 = d1.groupby('Bucket', as_index=True, observed=False)
    
    # Calculate WOE and IV for the bins
    d3 = _calculate_woe_iv(d2, justmiss)
    
    return d3


@validate_inputs
def char_bin(Y, X):
    """
    WHAT: Handles categorical variables by treating each category as a bin
    
    HOW IT WORKS:
    1. Each unique category becomes its own bin
    2. Missing values are treated as a separate category
    3. Calculates WOE/IV for each category
    
    WHY SEPARATE FROM CONTINUOUS:
    - Categories don't have natural order (e.g., Red, Blue, Green)
    - Can't apply monotonic binning concept
    - Each category gets its own WOE value
    
    EXAMPLE:
    If ProductCategory has values [A, B, C], we get:
    - WOE for category A
    - WOE for category B  
    - WOE for category C
    - WOE for missing values (if any)
    """
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X', 'Y']][df1.X.isnull()]
    notmiss = df1[['X', 'Y']][df1.X.notnull()]
    
    if len(notmiss) == 0:
        logger.warning("All values are missing for this variable")
        return _create_empty_woe_df()
    
    # Group by each unique category
    df2 = notmiss.groupby('X', as_index=True)
    
    # Calculate WOE/IV using same logic as continuous variables
    d3 = _calculate_woe_iv(df2, justmiss)
    
    # For categorical, MIN and MAX are the same (the category name)
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    
    return d3


def _calculate_woe_iv(grouped_data, missing_data=None):
    """
    WHAT: Core calculation engine for WOE and IV
    
    MATHEMATICAL FOUNDATION:
    
    Weight of Evidence (WOE):
    WOE = ln(Distribution of Events / Distribution of Non-Events)
    
    - Positive WOE: This bin has higher event rate than population
    - Negative WOE: This bin has lower event rate than population
    - WOE = 0: This bin has same event rate as population
    
    Information Value (IV):
    IV = Σ (Distribution of Events - Distribution of Non-Events) × WOE
    
    - Measures the predictive power of the entire variable
    - Sum of contributions from all bins
    
    STEP-BY-STEP CALCULATION:
    1. Count events and non-events in each bin
    2. Calculate distribution percentages
    3. Apply WOE formula with log transformation
    4. Calculate IV contribution for each bin
    5. Sum IV contributions for total IV
    
    HANDLING EDGE CASES:
    - If a bin has no events: WOE = -∞ (capped at large negative)
    - If a bin has no non-events: WOE = +∞ (capped at large positive)
    - Small epsilon added to prevent log(0)
    """
    d3 = pd.DataFrame()
    d3["COUNT"] = grouped_data.count().Y
    d3["EVENT"] = grouped_data.sum().Y

    try:
        # For continuous bins, the min/max of the original X values are needed
        d3["MIN_VALUE"] = grouped_data.min().X
        d3["MAX_VALUE"] = grouped_data.max().X
    except AttributeError:
        # For categorical, the value is just the index (the category name)
        d3["MIN_VALUE"] = d3.index
        d3["MAX_VALUE"] = d3.index
    
    d3["NONEVENT"] = d3["COUNT"] - d3["EVENT"]
    d3 = d3.reset_index(drop=True)
    
    # Add missing values as separate bin if they exist
    if missing_data is not None and len(missing_data.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE': np.nan}, index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = missing_data.count().Y
        d4["EVENT"] = missing_data.sum().Y
        d4["NONEVENT"] = d4["COUNT"] - d4["EVENT"]
        d3 = pd.concat([d3, d4], ignore_index=True)
    
    # Calculate totals for distribution
    total_events = d3['EVENT'].sum()
    total_nonevents = d3['NONEVENT'].sum()
    
    # Handle edge case: all observations in one class
    if total_events == 0 or total_nonevents == 0:
        logger.warning("No events or non-events in data. WOE/IV calculation may be unreliable.")
        d3["EVENT_RATE"] = 0
        d3["NON_EVENT_RATE"] = 0
        d3["DIST_EVENT"] = 0
        d3["DIST_NON_EVENT"] = 0
        d3["WOE"] = 0
        d3["IV"] = 0
    else:
        # Calculate event rates within each bin
        d3["EVENT_RATE"] = d3['EVENT'] / d3["COUNT"]
        d3["NON_EVENT_RATE"] = d3['NONEVENT'] / d3["COUNT"]
        
        # Calculate distributions with epsilon to prevent log(0)
        epsilon = 1e-10  # Very small number
        d3["DIST_EVENT"] = np.maximum(d3['EVENT'] / total_events, epsilon)
        d3["DIST_NON_EVENT"] = np.maximum(d3['NONEVENT'] / total_nonevents, epsilon)
        
        # WEIGHT OF EVIDENCE CALCULATION
        # ln(probability of being in this bin given event / probability given non-event)
        d3["WOE"] = np.log(d3["DIST_EVENT"] / d3["DIST_NON_EVENT"])
        
        # INFORMATION VALUE CALCULATION
        # Measures the "distance" between event and non-event distributions
        d3["IV"] = (d3["DIST_EVENT"] - d3["DIST_NON_EVENT"]) * d3["WOE"]
    
    d3["VAR_NAME"] = "VAR"
    
    # Reorder columns for readability
    d3 = d3[['VAR_NAME', 'MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 
             'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT', 'DIST_NON_EVENT', 'WOE', 'IV']]
    
    # Replace infinities with 0
    d3 = d3.replace([np.inf, -np.inf], 0)
    
    # Total IV is sum of all bin contributions
    d3["IV"] = d3["IV"].fillna(0)
    total_iv = d3["IV"].sum()
    d3.loc[:, "IV"] = total_iv  # Assign total to all rows
    
    return d3

def _create_empty_woe_df():
    """
    WHAT: Creates empty dataframe structure when no valid data exists
    
    WHY NEEDED:
    - Maintains consistent output format
    - Prevents downstream errors
    - Allows pipeline to continue with other variables
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
    """
    WHAT: Main function that calculates WOE/IV for all variables in a dataset
    
    HOW IT WORKS:
    1. Loops through each column in the dataframe
    2. Automatically detects variable type (continuous vs categorical)
    3. Applies appropriate binning method
    4. Aggregates results into summary tables
    
    AUTOMATIC VARIABLE TYPE DETECTION:
    - Numeric with >2 unique values → Continuous (monotonic binning)
    - Everything else → Categorical (character binning)
    
    OUTPUT INTERPRETATION:
    Returns two dataframes:
    1. Detailed WOE table: Shows WOE for each bin of each variable
    2. Summary IV table: Shows total IV for each variable with interpretation
    
    IV INTERPRETATION GUIDE:
    < 0.02: Not Predictive (drop these features)
    0.02-0.1: Weak Predictive Power
    0.1-0.3: Medium Predictive Power (good features)
    0.3-0.5: Strong Predictive Power (very good features)
    > 0.5: Suspicious (might be overfitting)
    
    EXAMPLE USAGE:
    detailed_woe, summary_iv = data_vars(df, df['PurchaseStatus'])
    """
    # Update global parameters if provided
    global DEFAULT_MAX_BIN, DEFAULT_FORCE_BIN
    if max_bin is not None:
        DEFAULT_MAX_BIN = max_bin
    if force_bin is not None:
        DEFAULT_FORCE_BIN = force_bin
    
    # Validate inputs
    if len(df1) != len(target):
        raise ValueError("DataFrame and target must have the same length")
    
    # EXTRACT TARGET VARIABLE NAME
    # This clever code looks at how the function was called
    # to automatically exclude the target from processing
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
            # VARIABLE TYPE DETECTION
            if pd.api.types.is_numeric_dtype(df1[i]) and df1[i].nunique() > 2:
                # Continuous variable: use monotonic binning
                conv = mono_bin(target, df1[i])
            else:
                # Categorical variable: use character binning
                conv = char_bin(target, df1[i])
            
            conv["VAR_NAME"] = i
            count = count + 1
            
            # Concatenate results
            if count == 0:
                iv_df = conv
            else:
                iv_df = pd.concat([iv_df, conv], ignore_index=True)
                
        except Exception as e:
            logger.error(f"Error processing variable {i}: {e}")
            continue  # Skip failed variables instead of crashing
    
    # Handle case where no variables were processed
    if iv_df is None:
        logger.warning("No variables were successfully processed")
        return pd.DataFrame(), pd.DataFrame()
    
    # CREATE SUMMARY TABLE
    # Shows total IV for each variable
    iv = pd.DataFrame({'IV': iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    
    # Sort by IV value (most predictive first)
    iv = iv.sort_values('IV', ascending=False)
    
    # ADD INTERPRETATION
    # Helps users understand which features to keep
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

"""
=================================================================================
END OF ANNOTATED CODE

KEY LEARNING POINTS FOR STUDENTS:

1. WOE TRANSFORMATION:
   - Converts any variable into a measure of its relationship with the target
   - Makes all variables comparable on the same scale
   - Handles missing values naturally

2. INFORMATION VALUE:
   - Single number that quantifies predictive power
   - Industry standard in financial modeling
   - Helps with automatic feature selection

3. MONOTONIC BINNING:
   - Creates interpretable bins for continuous variables
   - Ensures logical relationship (e.g., higher age → higher/lower purchase rate)
   - Reduces noise while preserving signal

4. PRACTICAL APPLICATIONS:
   - Credit scoring (predicting loan defaults)
   - Marketing (predicting customer response)
   - Risk modeling (predicting insurance claims)
   - Any binary classification in finance

5. ADVANTAGES OVER OTHER METHODS:
   - Handles non-linear relationships
   - No assumptions about distributions
   - Interpretable results
   - Works with small datasets

This module demonstrates how financial industry techniques can be
implemented in Python for practical machine learning applications.
=================================================================================
"""