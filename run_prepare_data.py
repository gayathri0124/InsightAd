from preprocessing.prepare_ctr_data import load_and_merge_data, preprocess_data

if __name__ == "__main__":
    print("Starting CTR data preparation pipeline...\n")

    df = load_and_merge_data()
    X, y = preprocess_data(df)

    print("\n Sample Features:")
    print(X.head(10))

    print("\n🔍 Class balance:")
    print(y.value_counts())

    print("\n Label Distribution(Ratio Clicked):")
    print(y.value_counts(normalize=True))


