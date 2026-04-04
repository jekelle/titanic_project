from pathlib import Path

import pandas as pd
import plotly.express as px

DATA_PATH = Path(__file__).resolve().parent / "titanic.csv"
df_titanic = pd.read_csv(DATA_PATH)


def survival_demographics():
    df = df_titanic.copy()

    pclass_order = [1, 2, 3]
    sex_order = ["female", "male"]
    age_order = ["Child", "Teen", "Adult", "Senior"]

    df["Pclass"] = pd.Categorical(df["Pclass"], categories=pclass_order, ordered=True)
    df["Sex"] = pd.Categorical(df["Sex"], categories=sex_order, ordered=True)

    df["age_group"] = pd.cut(
        df["Age"],
        bins=[0, 12, 19, 59, 120],
        labels=age_order,
        include_lowest=True,
        right=True,
        ordered=True,
    )

    summary = (
        df.groupby(["Pclass", "Sex", "age_group"], observed=False)
        .agg(
            n_passengers=("Survived", "size"),
            n_survivors=("Survived", "sum"),
        )
    )

    full_index = pd.MultiIndex.from_product(
        [pclass_order, sex_order, age_order],
        names=["Pclass", "Sex", "age_group"],
    )

    summary = summary.reindex(full_index, fill_value=0).reset_index()
    summary["age_group"] = pd.Categorical(
        summary["age_group"], categories=age_order, ordered=True
    )

    summary["survival_rate"] = summary["n_survivors"] / summary["n_passengers"]
    summary.loc[summary["n_passengers"] == 0, "survival_rate"] = pd.NA

    return summary


def visualize_demographic():
    summary = survival_demographics()
    return px.bar(
        summary,
        x="Pclass",
        y="survival_rate",
        color="Sex",
        facet_col="age_group",
        barmode="group",
        title="Survival Rate by Class, Sex, and Age Group",
    )


def family_groups():
    df = df_titanic.copy()
    df["family_size"] = df["SibSp"] + df["Parch"] + 1

    summary = (
        df.groupby(["Pclass", "family_size"], observed=False)
        .agg(
            n_passengers=("PassengerId", "count"),
            avg_fare=("Fare", "mean"),
            min_fare=("Fare", "min"),
            max_fare=("Fare", "max"),
        )
        .reset_index()
        .sort_values(["Pclass", "family_size"], kind="stable")
        .reset_index(drop=True)
    )

    return summary


def last_names():
    df = df_titanic.copy()
    df["last_name"] = df["Name"].astype(str).str.split(",").str[0].str.strip()
    return df["last_name"].value_counts()


def visualize_families():
    summary = family_groups()
    return px.scatter(
        summary,
        x="family_size",
        y="avg_fare",
        color="Pclass",
        size="n_passengers",
        title="Family Size vs Average Fare by Passenger Class",
        labels={
            "family_size": "Family Size",
            "avg_fare": "Average Fare",
        },
    )
