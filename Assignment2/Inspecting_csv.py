import pandas as pd


CSV_PATH = "dd-translation-samples-aug26-small.csv"


def extract_n(df: pd.DataFrame) -> pd.Series:
    """Return the training-set size column ``n`` from a samples dataframe."""
    return df["n"]


def load_samples(path: str = CSV_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


if __name__ == "__main__":
    n = extract_n(load_samples())
    print(n.head())
