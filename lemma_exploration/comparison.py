from lemma_exploration.dataset import load_dataset

if __name__ == "__main__":
    name1, config1, split1 = (
        "yalhessi/lemexp-results",
        "default",
        # "lemexp-hol+afp20-opt-1.3b",
        "test",
    )
    name2, config2, split2 = (
        "yalhessi/lemexp-results",
        "default",
        # "lemexp-hol-opt-1.3b",
        "test",
    )
    results1 = load_dataset(name1, config1, split1, preprocess=False)
    results2 = load_dataset(name2, config2, split2, preprocess=False)

    passes_generation1, passes_generation2 = (
        results1["passes_generation"],
        results2["passes_generation"],
    )
    passes_syntax1, passes_syntax2 = (
        results1["passes_syntax"],
        results2["passes_syntax"],
    )
    passes_counterexample1, passes_counterexample2 = (
        results1["passes_counterexample"],
        results2["passes_counterexample"],
    )

    # make a bar chart comparing the results
    # import necessary libraries
    # import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    # create DataFrame
    df = pd.DataFrame(
        {
            "True": [28, 30, 34],
            "False": [22, 26, 30],
        },
        index=[
            "Passes Generation",
            "Passes Syntax",
            "Passes Counterexample",
        ],
    )

    # create stacked bar chart for monthly temperatures
    df.plot(kind="bar", stacked=True, color=["green", "red"])

    # labels for x & y axis
    plt.xlabel("Metrics")
    plt.ylabel("Count")

    # title of plot
    plt.title("Model Evaluation")
