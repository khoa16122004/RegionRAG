from datasets import load_dataset

ds = load_dataset(
    "openbmb/VisRAG-Ret-Train-In-domain-data",
)

ds.save_to_disk("./VisRAG-Ret-Train-In-domain-data")


ds = load_dataset(
    "deepcs233/Visual-CoT",
)
ds.save_to_disk("./Visual-CoT")