cat > src/utils.py << 'EOF'
import pandas as pd

def load_data(path, sep=';'):
    """
    Loads a dataset using the given separator.
    Default separator is ';' since UCI student data uses semicolons.
    """
    df = pd.read_csv(path, sep=sep)
    return df
EOF
