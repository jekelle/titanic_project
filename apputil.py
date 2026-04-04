import pandas as pd
import plotly.express as px


def survival_demographics(df):
    df = df.copy()
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 19, 59, 120],
        labels=["Child", "Teen", "Adult", "Senior"]
    )
    grouped = df.groupby(["Pclass", "Sex", "AgeGroup"])
    summary = grouped.agg(
        n_passengers=("Survived", "count"),
        n_survivors=("Survived", "sum")
    ).reset_index()
    summary["survival_rate"] = summary["n_survivors"] / summary["n_passengers"]
    summary = summary.sort_values(by=["Pclass", "Sex", "AgeGroup"])
    return summary


def visualize_demographic(summary):
    fig = px.bar(
        summary,
        x="Pclass",
        y="survival_rate",
        color="Sex",
        facet_col="AgeGroup",
        barmode="group",
        title="Survival Rate by Class, Sex, and Age Group"
    )
    return fig


def family_groups(df):
    df = df.copy()
    df["family_size"] = df["SibSp"] + df["Parch"] + 1
    grouped = df.groupby(["Pclass", "family_size"])
    summary = grouped.agg(
        n_passengers=("PassengerId", "count"),
        avg_fare=("Fare", "mean"),
        min_fare=("Fare", "min"),
        max_fare=("Fare", "max")
    ).reset_index()
    summary = summary.sort_values(by=["Pclass", "family_size"])
    return summary


def last_names(df):
    df = df.copy()
    df["last_name"] = df["Name"].str.split(",").str[0]
    return df["last_name"].value_counts()


def visualize_families(summary):
    fig = px.scatter(
        summary,
        x="family_size",
        y="avg_fare",
        color="Pclass",
        size="n_passengers",
        title="Family Size vs Average Fare by Passenger Class",
        labels={
            "family_size": "Family Size",
            "avg_fare": "Average Fare"
        }
    )
    return fig
