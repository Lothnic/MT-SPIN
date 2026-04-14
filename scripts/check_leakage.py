import pandas as pd

def main():
    print("Loading datasets...")
    train_df = pd.read_parquet("data/processed/train.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")
    
    train_hin = set(train_df["translation"].apply(lambda x: x["hin_Deva"]))
    train_kng = set(train_df["translation"].apply(lambda x: x["kng_Deva"]))
    
    test_hin = set(test_df["translation"].apply(lambda x: x["hin_Deva"]))
    test_kng = set(test_df["translation"].apply(lambda x: x["kng_Deva"]))
    
    hin_leakage = train_hin.intersection(test_hin)
    kng_leakage = train_kng.intersection(test_kng)
    
    print(f"Hindi sequence leakages: {len(hin_leakage)}")
    print(f"Kangri sequence leakages: {len(kng_leakage)}")
    
    if len(hin_leakage) > 0 or len(kng_leakage) > 0:
        print("Filtering out leakage from test set...")
        clean_test = test_df[~test_df["translation"].apply(
            lambda x: x["hin_Deva"] in train_hin or x["kng_Deva"] in train_kng
        )]
        print(f"Original test size: {len(test_df)}, Cleaned test size: {len(clean_test)}")
        clean_test.to_parquet("data/processed/test.parquet")
        print("Cleaned test.parquet saved.")
    else:
        print("No leakage detected between train and test datasets.")

if __name__ == "__main__":
    main()
