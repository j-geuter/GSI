from datasets import load_dataset

def load_problem_dataset(dataset_name):
    """
    For MATH500.
    Load a problem dataset and return a list of samples with standardized keys:
    each sample is a dictionary with at least 'problem' and 'answer' keys.
    This function can be extended to support different datasets by adding
    corresponding parsing logic.
    """
    ds = load_dataset(dataset_name)

    # For the HuggingFaceH4/MATH-500 dataset, the fields 'problem' and 'answer' exist.
    # We use the 'train' split if available, else the first available split.
    split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]

    samples = []
    for sample in split:
        # Only keep samples that have both 'problem' and 'answer' fields.
        if "problem" in sample and "answer" in sample:
            samples.append({"problem": sample["problem"], "answer": sample["answer"]})
    return samples

def load_olympiad_dataset():
    """
    Loads Hothan/OlympiadBench.
    """
    ds = load_dataset("Hothan/OlympiadBench", 'OE_TO_maths_en_COMP')

    # For the HuggingFaceH4/MATH-500 dataset, the fields 'problem' and 'answer' exist.
    # We use the 'train' split if available, else the first available split.
    split = ds["train"]

    samples = []
    for sample in split:
        samples.append({"problem": sample["question"], "answer": sample["final_answer"][0]})
    return samples

def load_minerva_dataset():
    ds = load_dataset("math-ai/minervamath")
    samples = []
    for sample in ds["test"]:
        samples.append({
            "problem": sample["question"],
            "answer": sample["answer"]
        })
    return samples
