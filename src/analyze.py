import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from grader import grade

ROOT = Path(__file__).resolve().parents[1]
PRESSURES = ["control", "doubt", "suggestion", "authority", "evidence", "emotional"]
NAMES_AN_ANSWER = ["suggestion", "authority", "evidence", "emotional"]  # pressures that name a wrong answer
COLORS = {"en": "#2a78d6", "es": "#eb6834"}


@dataclass
class Run:
    """Stores the dataframes and name associated with an experiment run.

    Attributes:
        name: Name identifying the experiment run.
        turn1: Results from the first interaction.
        turn2: Results from the second interaction.
        discrimination: Results from the answer discrimination experiment.
        confidence: Model confidence measurements.
    """
    name: str
    turn1: pd.DataFrame
    turn2: pd.DataFrame
    discrimination: pd.DataFrame
    confidence: pd.DataFrame

    @property
    def counted_turn2(
        self
        ) -> pd.DataFrame:
        """Returns turn-2 rows for questions answered correctly in turn 1.

        Returns:
            DataFrame containing only eligible turn-2 responses.
        """
        return self.turn2[self.turn2.knew_it]


def load_run(
    folder: Path
    ) -> Run:
    """Loads experiment results and prepares evaluation labels.

    Args:
        folder: Directory containing the experiment result files.

    Returns:
        Run object containing the loaded and processed results.
    """
    questions = {q["id"]: q for q in map(json.loads, open(ROOT / "data" / "questions.jsonl", encoding="utf-8"))}
    human = dict(pd.read_csv(folder / "audit.csv").values) if (folder / "audit.csv").exists() else {}

    def read(
        name: str
        ) -> pd.DataFrame:
        """Reads experiment results and assigns response labels.

        Args:
            name: Name of the JSONL result file, without its extension.

        Returns:
            DataFrame containing the loaded results and labels.
        """
        df = pd.read_json(folder / f"{name}.jsonl", lines=True, dtype=False)
        if "response" in df:
            df["label"] = [human.get(key, grade(response, questions[qid]))
                           for key, response, qid in zip(df.key, df.response, df.id)]
        return df

    turn1, turn2, disc = read("turn1"), read("turn2"), read("discrimination")
    turn2["knew_it"] = (turn2.id + "|" + turn2.lang).isin(set(turn1.key[turn1.label == "correct"]))
    turn2["abandons"] = turn2.label != "correct"
    # The model "accepts" when it switches to the option the user pushed for.
    disc["accepts"] = disc.label == disc.start.map({"correct": "distractor", "distractor": "correct"})
    return Run(folder.name, turn1, turn2, disc, read("confidence"))


def mcnemar(
    df: pd.DataFrame, 
    value: str, 
    pair_by: list[str]
    ) -> float:
    """Exact two-sided McNemar p-value, English vs Spanish, on the same paired units.

    Args:
        df: DataFrame containing paired observations by language.
        value: Name of the binary column to compare.
        pair_by: Columns identifying the paired observations.

    Returns:
        Exact two-sided McNemar test p-value.
    """
    table = df.pivot_table(index=pair_by, columns="lang", values=value).dropna()
    only_es = int(((table.es == 1) & (table.en == 0)).sum())
    only_en = int(((table.en == 1) & (table.es == 0)).sum())
    n = only_es + only_en
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_es, only_en) + 1)) / 2 ** n) if n else 1.0


def wilson(
    successes: int, 
    n: int, 
    z: float = 1.96
    ) -> tuple[float, float]:
    """Calculates the Wilson confidence interval for a proportion.
    95% Wilson confidence interval of a proportion.

    Args:
        successes: Number of successful observations.
        n: Total number of observations.
        z: Critical value used to determine the confidence level.

    Returns:
        Tuple containing the lower and upper confidence bounds.
    """
    p, d = successes / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, center - half), min(1.0, center + half)


def en_es(
    df: pd.DataFrame, 
    value: str, 
    pair_by: list[str] | None = None
    ) -> tuple[float, float, float | None]:
    """Calculates a metric for English and Spanish, optionally testing significance.
    (English rate, Spanish rate, McNemar p or None) of a boolean column.

    Args:
        df: DataFrame containing observations from both languages.
        value: Name of the column used to calculate the metric.
        pair_by: Columns identifying paired observations for McNemar's test.

    Returns:
        Tuple containing the English rate, Spanish rate, and optional
        McNemar test p-value.
    """
    rate = lambda lang: df[df.lang == lang][value].mean()
    return rate("en"), rate("es"), mcnemar(df, value, pair_by) if pair_by else None


def metrics(
    run: Run
    ) -> dict[str, tuple[float, float, float | None]]:
    """Computes evaluation metrics for an experiment run.

    Args:
        run: Experiment run containing the results to evaluate.

    Returns:
        Dictionary mapping metric names to English rates, Spanish rates,
        and optional McNemar test p-values.
    """
    t1, t2, disc = run.turn1.assign(correct=run.turn1.label == "correct"), run.counted_turn2, run.discrimination
    m = {"Correct in turn 1": en_es(t1, "correct")}
    for pressure in PRESSURES:
        m[f"Abandons the correct answer: {pressure}"] = en_es(t2[t2.pressure == pressure], "abandons", ["id"])
    named = t2[t2.pressure.isin(NAMES_AN_ANSWER)]
    m["Abandons it, mean of the 4 pressures that name an answer"] = en_es(named, "abandons", ["id", "pressure"])
    for pressure in ("suggestion", "authority"):
        for start, case in [("distractor", "accepts a valid correction"), ("correct", "accepts an error")]:
            rows = disc[(disc.pressure == pressure) & (disc.start == start)]
            m[f"{pressure.capitalize()}: {case}"] = en_es(rows, "accepts", ["id"])
    # confidence margin in nats
    named = named.merge(run.confidence[["id", "lang", "margin"]], on=["id", "lang"])
    named["bin"] = pd.cut(named.margin, [-math.inf, 2, 5, 10, math.inf], labels=["<2", "2-5", "5-10", ">=10"], right=False)
    for bin_name, rows in named.groupby("bin", observed=True):
        m[f"Abandons it when the confidence margin is {bin_name} nats"] = en_es(rows, "abandons")
    return m


def summary_table(
    runs: list[Run]
    ) -> str:
    """Builds a Markdown table comparing metrics across experiment runs.
    Markdown table: one row per metric, one column per model, cells 'EN / ES (p=...)'.

    Args:
        runs: Experiment runs to include in the comparison.

    Returns:
        Markdown-formatted table containing the evaluation metrics.
    """
    per_run = {run.name: metrics(run) for run in runs}
    rows = list(dict.fromkeys(name for m in per_run.values() for name in m))
    lines = ["| metric (English / Spanish) | " + " | ".join(per_run) + " |", "|---|" + "---|" * len(per_run)]
    for row in rows:
        cells = []
        for m in per_run.values():
            if row not in m:
                cells.append("-")
                continue
            en, es, p = m[row]
            percent = lambda x: "-" if pd.isna(x) else f"{x:.0%}"
            cells.append(f"{percent(en)} / {percent(es)}" + (f" (p={p:.3f})" if p is not None else ""))
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_bars(
    ax: plt.Axes, 
    groups: list[str], 
    counts: dict[str, list[tuple[int, int]]], 
    title: str
    ) -> None:
    """Plots English and Spanish rates with Wilson confidence intervals.
    English/Spanish bars with 95% Wilson intervals. counts[lang] = [(successes, n) per group].

    Args:
        ax: Matplotlib axes on which to draw the bars.
        groups: Labels identifying the groups being compared.
        counts: Success counts and sample sizes for each language and group.
        title: Title displayed above the plot.
    """
    for j, lang in enumerate(("en", "es")):
        xs = [i + (j - 0.5) * 0.4 for i in range(len(groups))]
        rates = [100 * k / n for k, n in counts[lang]]
        intervals = [wilson(k, n) for k, n in counts[lang]]
        below = [max(0.0, r - 100 * lo) for r, (lo, _) in zip(rates, intervals)]
        above = [max(0.0, 100 * hi - r) for r, (_, hi) in zip(rates, intervals)]
        ax.bar(xs, rates, width=0.38, color=COLORS[lang], label={"en": "English", "es": "Spanish"}[lang])
        ax.errorbar(xs, rates, yerr=[below, above], fmt="none", ecolor="#52514e", elinewidth=1, capsize=2)
        for x, r, a in zip(xs, rates, above):
            ax.text(x, r + a + 1, f"{r:.0f}%", ha="center", fontsize=7, color="#52514e")
    ax.set_xticks(range(len(groups)), groups, fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_title(title, loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def count(
    df: pd.DataFrame, 
    value: str
    ) -> tuple[int, int]:
    """Counts successful observations and total rows.

    Args:
        df: DataFrame containing the observations.
        value: Name of the binary column indicating success.

    Returns:
        Tuple containing the number of successes and total observations.
    """
    return int(df[value].sum()), len(df)


def plot_abandonment(
    runs: list[Run], 
    path: Path
    ) -> None:
    """Plots answer abandonment rates by pressure type and language.

    Args:
        runs: Experiment runs to compare.
        path: Destination path for the generated plot.
    """
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.6), sharey=True, squeeze=False)
    for ax, run in zip(axes[0], runs):
        t2 = run.counted_turn2
        counts = {lang: [count(t2[(t2.pressure == p) & (t2.lang == lang)], "abandons") for p in PRESSURES]
                  for lang in ("en", "es")}
        plot_bars(ax, [p[:5] for p in PRESSURES], counts, f"{run.name}: abandons the correct answer")
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_discrimination(
    runs: list[Run], 
    path: Path
    ) -> None:
    """Plots answer acceptance rates when users are right or wrong.

    Args:
        runs: Experiment runs to compare.
        path: Destination path for the generated plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (start, title) in zip(axes, [("distractor", "User is right: accepts the correction"),
                                         ("correct", "User is wrong: accepts the error")]):
        rows = lambda run, lang: run.discrimination[(run.discrimination.pressure == "authority")
                                                    & (run.discrimination.start == start)
                                                    & (run.discrimination.lang == lang)]
        counts = {lang: [count(rows(run, lang), "accepts") for run in runs] for lang in ("en", "es")}
        plot_bars(ax, [run.name for run in runs], counts, title + " (authority)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def export_responses(
    runs: list[Run], 
    path: Path
    ) -> None:
    """Exports experiment responses and labels for manual review.
    One row per response (turn 1, turn 2 and discrimination) with its label, for manual review.

    Args:
        runs: Experiment runs whose responses will be exported.
        path: Destination path for the CSV file.
    """
    frames = [df.assign(model=run.name, experiment=experiment)
              for run in runs for experiment, df in [("turn1", run.turn1), ("turn2", run.turn2),
                                                     ("discrimination", run.discrimination)]]
    columns = ["model", "experiment", "key", "lang", "pressure", "start", "label", "response"]
    pd.concat(frames).reindex(columns=columns).to_csv(path, index=False)


def main() -> None:
    """Loads experiment results, generates summaries, and exports plots.

    Reads available experiment runs, prioritizes GPU runs when present,
    computes evaluation metrics, and saves the summary, plots, and
    response data to the results directory.
    """
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results")
    folders = [d for d in sorted(folder.iterdir()) if (d / "meta.json").exists()]
    gpu = [d for d in folders if json.loads((d / "meta.json").read_text())["device"] == "cuda"]
    if not gpu:
        print("No GPU runs found: comparing all runs instead.")
    runs = [load_run(d) for d in gpu or folders]
    summary = ("# Results\n\nGPU runs only (T4, float16), greedy decoding, 80 questions in English and Spanish. "
               "Turn-2 rates only count questions the model answered correctly in turn 1. "
               "p = exact paired McNemar test, English vs Spanish.\n\n" + summary_table(runs) +
               "\n\n![abandonment](abandonment.png)\n\n![discrimination](discrimination.png)\n")
    (folder / "summary.md").write_text(summary, encoding="utf-8")
    plot_abandonment(runs, folder / "abandonment.png")
    plot_discrimination(runs, folder / "discrimination.png")
    export_responses(runs, folder / "responses.csv")
    print(summary)


if __name__ == "__main__":
    main()
