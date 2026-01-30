########################
#Libraries
########################

import pandas as pd
import pathlib 
from io import StringIO
import numpy as np
import re
import hashlib


########################
#Data Read in function 
########################

def read_df(input):
    if isinstance(input, pd.DataFrame):
        return input
    
    try:
        path = pathlib.Path(input)

    except TypeError:
        raise TypeError("Input must be a DataFrame or a path to a CSV file")

    if path.is_file() and path.suffix == ".csv":
        return pd.read_csv(path)

    print("Error. Try pass a CSV direct or Filepath to a CSV.")
    return None



########################
#df checker 
#(should check for empty data frames and do some basic data Anonymisation)
########################
'''
def df_checker(data):
    #shape
    shape = data.shape
    #column names
    col_names = data.columns.tolist()

    #info as string
    buffer = StringIO()
    data.info(buf=buffer)
    info = buffer.getvalue()

    #missing values
    per_null= data.isna().mean() * 100
    num_null = data.isna().sum()



    #categorical columns
    cat_cols = data.select_dtypes(include='object').columns

    unique_val = {col: data[col].nunique() for col in cat_cols}
    num_cat_vals = {col: data[col].value_counts() for col in cat_cols}
    cat_col_proportion = {col: data[col].value_counts(normalize=True) for col in cat_cols}


    return {
            "shape": shape,
            "info": info,
            "percent_null": per_null,
            "num_null": num_null,
            "unique_values": unique_val,
            "cat_value_counts": num_cat_vals,
            "cat_value_proportion": cat_col_proportion,
            "col_names": col_names
        }
''' 




from io import StringIO
import pandas as pd
import numpy as np
'''
def df_checker(data: pd.DataFrame) -> dict:
    # --- basic shape / columns ---
    shape = data.shape
    col_names = data.columns.tolist()

    # --- column type summary ---
    dtype_counts = data.dtypes.value_counts().to_dict()
    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    cat_cols = data.select_dtypes(include="object").columns.tolist()

    # --- missing values (signals only) ---
    per_null = data.isna().mean() * 100
    cols_high_missing = per_null[per_null > 10].round(2).to_dict()
    cols_no_missing = per_null[per_null == 0].index.tolist()

    # --- categorical column signals ---
    cat_summary = {}

    for col in cat_cols:
        nunique = data[col].nunique(dropna=True)
        total = len(data[col])

        top_freq = data[col].value_counts(normalize=True, dropna=True).iloc[0] if total > 0 else 0

        cat_summary[col] = {
            "unique_values": int(nunique),
            "high_cardinality": bool(nunique > 50),
            "low_cardinality": bool(nunique <= 10),
            "top_value_proportion": round(float(top_freq), 3),
        }

    #numeric column signals
    num_summary = {}

    for col in numeric_cols:
        series = data[col].dropna()
        if series.empty:
            continue

        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1

        outlier_ratio = (
            ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).mean()
            if iqr > 0
            else 0
        )

        num_summary[col] = {
            "outlier_ratio": round(float(outlier_ratio), 3),
            "skewed": bool(abs(series.skew()) > 1),
        }

    #quality issues
    potential_issues = {}
    for col in cat_cols:
        col_data = data[col].dropna().astype(str)
        if col_data.empty:
            continue
        
        potential_issues[col] = {
            "has_leading_trailing_whitespace": bool((col_data.str.strip() != col_data).any()),
            "mixed_case": bool(col_data.str.islower().any() and col_data.str.isupper().any())
        }

    #numeric columns that might be categorical
    numeric_potentially_categorical = [
        col for col in numeric_cols 
        if data[col].nunique() < 20
    ]

    # --- potential date columns
    potential_date_columns = [
        col for col in cat_cols
        if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'founded', 'created', 'updated'])
    ]

    #duplicate rows check
    duplicate_info = {
        "count": int(data.duplicated().sum()),
        "percentage": round(float(data.duplicated().mean() * 100), 2)
    }

    #Helper function for pattern extraction
    def get_value_pattern(val):
        """Extract pattern info without exposing actual content"""
        val_str = str(val)
        
        return {
            "length": len(val_str),
            "word_count": len(val_str.split()),
            "has_digits": any(c.isdigit() for c in val_str),
            "has_letters": any(c.isalpha() for c in val_str),
            "has_special_chars": any(not c.isalnum() and not c.isspace() for c in val_str),
            "all_caps": val_str.isupper() if val_str else False,
            "title_case": val_str.istitle() if val_str else False,
            "contains_currency": any(sym in val_str for sym in ['$', '€', '£', '¥']),
            "contains_comma": ',' in val_str,
            "contains_hyphen": '-' in val_str,
            "contains_parentheses": '(' in val_str or ')' in val_str,
        }

    # --- example patterns (anonymized - no actual data exposed) ---
    example_patterns = {
        col: [get_value_pattern(v) for v in data[col].dropna().head(3)]
        for col in data.columns
    }

    return {
        "shape": shape,
        "column_names": col_names,
        "dtype_counts": dtype_counts,
        "missing_values": {
            "columns_high_missing_pct": cols_high_missing,
            "columns_no_missing": cols_no_missing,
        },
        "categorical_summary": cat_summary,
        "numeric_summary": num_summary,
        "potential_issues": potential_issues,
        "numeric_potentially_categorical": numeric_potentially_categorical,
        "potential_date_columns": potential_date_columns,
        "duplicate_rows": duplicate_info,
        "example_patterns": example_patterns, 
    }
#anonymise data for security purposes
#DO NOT SEND SENSITIVE DATA INTO AN LLM EVER! I AM NOT LIABLE IF YOU DO THAT!
#WIP 
'''

########################
#Data Processing Prompt + Constraints
#(you can build registries in here that help guide the LLM)
########################




########################
#Prompt builder
#(this is where the registries and the prompt will be created)
########################


########################
#API Client call
########################


########################
#Main function to call
########################
import pandas as pd
import numpy as np
import re
import hashlib




def get_structure_pattern(val):
    """
    Extract structural characteristics without exposing content.
    
    This reveals FORMAT issues without revealing DATA.
    """
    if pd.isna(val):
        return None
    
    val_str = str(val)
    
    # Create a structure representation
    structure = []
    i = 0
    while i < len(val_str):
        char = val_str[i]
        if char.isalpha():
            # Count consecutive letters
            count = 1
            while i + count < len(val_str) and val_str[i + count].isalpha():
                count += 1
            structure.append(f"ALPHA({count})")
            i += count
        elif char.isdigit():
            # Count consecutive digits
            count = 1
            while i + count < len(val_str) and val_str[i + count].isdigit():
                count += 1
            structure.append(f"DIGIT({count})")
            i += count
        elif char in ['\n', '\t', '\r']:
            structure.append(f"NEWLINE" if char == '\n' else "TAB" if char == '\t' else "RETURN")
            i += 1
        elif char == ' ':
            structure.append("SPACE")
            i += 1
        else:
            structure.append(f"'{char}'")
            i += 1
    
    return '-'.join(structure[:20])  # Limit length


def df_checker(data: pd.DataFrame, sample_size: int = 5) -> dict:
    """
    Analyze data quality without exposing sensitive content.
    
    This function detects format issues while maintaining privacy by:
    1. Showing STRUCTURE, not content
    2. Detecting patterns algorithmically
    3. Using anonymized examples only when necessary
    """
    
    shape = data.shape
    col_names = data.columns.tolist()
    dtype_counts = data.dtypes.value_counts().to_dict()
    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    cat_cols = data.select_dtypes(include="object").columns.tolist()
    
    # Missing values
    per_null = data.isna().mean() * 100
    cols_high_missing = per_null[per_null > 10].round(2).to_dict()
    
    # === CATEGORICAL COLUMNS (PRIVACY-PRESERVING) ===
    cat_summary = {}
    
    for col in cat_cols:
        col_data = data[col].dropna().astype(str)
        if col_data.empty:
            continue
        
        nunique = data[col].nunique(dropna=True)
        
        # PRIVACY: Don't show actual values, show STRUCTURE
        # Instead of: ['Healthfirst\n3.1', 'ManTech\n4.2']
        # Show: ['ALPHA(11)-NEWLINE-DIGIT(1)-'.'-DIGIT(1)', ...]
        
        structure_samples = [
            get_structure_pattern(val) 
            for val in col_data.head(sample_size)
        ]
        
        # CRITICAL DETECTION: Embedded data (without exposing content)
        has_newlines = col_data.str.contains('\n', na=False).sum()
        has_tabs = col_data.str.contains('\t', na=False).sum()
        has_pipes = col_data.str.contains(r'\|', na=False).sum()
        has_semicolons = col_data.str.contains(';', na=False).sum()
        
        # Format detection (algorithmic - no content exposure)
        format_analysis = {
            "has_email_format": col_data.str.contains(r'[\w\.-]+@[\w\.-]+', na=False).sum(),
            "has_url_format": col_data.str.contains(r'https?://', na=False).sum(),
            "has_phone_format": col_data.str.contains(r'\d{3}[-.]?\d{3}[-.]?\d{4}', na=False).sum(),
            "has_date_format": col_data.str.contains(r'\d{4}[-/]\d{2}[-/]\d{2}', na=False).sum(),
            "has_currency_range": col_data.str.contains(r'\$\d+[KMB]?-\$\d+[KMB]?', na=False, regex=True).sum(),
            "has_number_range": col_data.str.contains(r'\d+-\d+', na=False).sum(),
            "has_comma_list": (col_data.str.count(',') > 1).sum(),
        }
        
        # Only include detected formats
        detected_formats = {k: v for k, v in format_analysis.items() if v > 0}
        
        # String length analysis
        str_lengths = col_data.str.len()
        
        # Character composition (reveals embedded data without showing it)
        char_analysis = {
            "avg_length": round(float(str_lengths.mean()), 1),
            "max_length": int(str_lengths.max()),
            "contains_digits_pct": round((col_data.str.contains(r'\d', na=False).sum() / len(col_data) * 100), 1),
            "contains_special_pct": round((col_data.str.contains(r'[^a-zA-Z0-9\s]', na=False).sum() / len(col_data) * 100), 1),
        }
        
        # Missing value indicators (no privacy concerns)
        missing_indicators = {
            "-1": int((data[col] == "-1").sum()),
            "Unknown": int((data[col] == "Unknown").sum()),
            "N/A": int((data[col] == "N/A").sum()),
            "null": int((data[col] == "null").sum()),
        }
        missing_indicators = {k: v for k, v in missing_indicators.items() if v > 0}
        
        cat_summary[col] = {
            "unique_count": int(nunique),
            "high_cardinality": bool(nunique > 50),
            
            # PRIVACY: Structure, not content
            "value_structure_examples": structure_samples,
            
            # CRITICAL: Embedded data detection (no content exposure)
            "embedded_data_signals": {
                "newlines": int(has_newlines),
                "tabs": int(has_tabs),
                "pipes": int(has_pipes),
                "semicolons": int(has_semicolons),
            },
            
            # Character composition analysis
            "character_analysis": char_analysis,
            
            # Detected formats
            "detected_formats": detected_formats,
            
            # Missing indicators
            "missing_indicators": missing_indicators,
        }
    
    # === NUMERIC COLUMNS (PRIVACY-PRESERVING) ===
    num_summary = {}
    
    for col in numeric_cols:
        series = data[col].dropna()
        if series.empty:
            continue
        
        # Negative value detection (no privacy concerns)
        negative_count = int((data[col] < 0).sum())
        actual_min = float(series.min())
        actual_max = float(series.max())
        
        # Discreteness check
        is_discrete = series.nunique() < 20
        
        # PRIVACY: Don't show actual values, show DISTRIBUTION
        # Instead of: [3.1, 4.2, 3.8, ...]
        # Show: statistical summary only
        
        num_summary[col] = {
            "range": {
                "min": actual_min,
                "max": actual_max,
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std()), 2),
            },
            
            # CRITICAL: Negative values (missing indicators)
            "negative_values": {
                "count": negative_count,
                "percentage": round(negative_count / len(data) * 100, 2),
                "likely_missing_indicator": bool(negative_count > 0 and actual_min == -1),
            },
            
            # Discreteness
            "potentially_categorical": bool(is_discrete),
            "unique_count": int(series.nunique()),
            
            # PRIVACY: No actual values shown
            "value_distribution": "See range and stats above"
        }
    
    # === CRITICAL ISSUES SUMMARY ===
    critical_issues = []
    
    for col in cat_cols:
        if col in cat_summary:
            # Flag embedded newlines
            if cat_summary[col]["embedded_data_signals"]["newlines"] > len(data) * 0.5:
                critical_issues.append({
                    "column": col,
                    "issue": "embedded_newlines",
                    "severity": "CRITICAL",
                    "description": f"{cat_summary[col]['embedded_data_signals']['newlines']} values ({cat_summary[col]['embedded_data_signals']['newlines']/len(data)*100:.1f}%) contain newlines",
                    "explanation": "This suggests multiple fields are concatenated in one column",
                    "example_structure": cat_summary[col]["value_structure_examples"][0] if cat_summary[col]["value_structure_examples"] else None,
                })
            
            # Flag range formats
            if "has_currency_range" in cat_summary[col]["detected_formats"]:
                count = cat_summary[col]["detected_formats"]["has_currency_range"]
                if count > len(data) * 0.5:
                    critical_issues.append({
                        "column": col,
                        "issue": "range_format",
                        "severity": "HIGH",
                        "description": f"{count} values ({count/len(data)*100:.1f}%) contain range format (e.g., $X-$Y)",
                        "explanation": "Cannot be directly converted to numeric - needs parsing into min/max",
                        "example_structure": "Format: $DIGIT-$DIGIT (estimated pattern)",
                    })
            
            # Flag comma-separated lists
            if "has_comma_list" in cat_summary[col]["detected_formats"]:
                count = cat_summary[col]["detected_formats"]["has_comma_list"]
                if count > len(data) * 0.3:
                    critical_issues.append({
                        "column": col,
                        "issue": "comma_separated_list",
                        "severity": "MEDIUM",
                        "description": f"{count} values contain multiple commas (likely lists)",
                        "explanation": "May need to be split into separate records or list column",
                    })
    
    for col in numeric_cols:
        if col in num_summary:
            # Flag negative values as missing indicators
            if num_summary[col]["negative_values"]["likely_missing_indicator"]:
                critical_issues.append({
                    "column": col,
                    "issue": "negative_missing_indicator",
                    "severity": "HIGH",
                    "description": f"{num_summary[col]['negative_values']['count']} negative values (min={num_summary[col]['range']['min']})",
                    "explanation": "Negative values (especially -1) likely represent missing data, not valid measurements",
                })
    
    return {
        "shape": shape,
        "column_names": col_names,
        "dtype_counts": dtype_counts,
        "missing_values": {
            "columns_high_missing_pct": cols_high_missing,
        },
        "categorical_summary": cat_summary,
        "numeric_summary": num_summary,
        "critical_issues": critical_issues,
        
        
        "privacy_note": "This analysis reveals data structure and format patterns without exposing actual content."
    }

