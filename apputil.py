import pandas as pd
import plotly.express as px

def survival_demographics(df):

    bins = [0,12,19,59,200]
    labels = ["Child","Teen","Adult","Senior"]

    df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels)

    table = df.groupby(["Pclass","Sex","age_group"], dropna=False).agg(
        n_passengers=("PassengerId","count"),
        n_survivors=("Survived","sum")
    )

    table["survival_rate"] = table["n_survivors"] / table["n_passengers"]

    return table.reset_index().sort_values(["Pclass","Sex","age_group"])


def visualize_demographic(table):

    fig = px.bar(
        table,
        x="age_group",
        y="survival_rate",
        color="Sex",
        facet_col="Pclass",
        barmode="group"
    )
    return fig


def family_groups(df):

    df["family_size"] = df["SibSp"] + df["Parch"] + 1

    table = df.groupby(["Pclass","family_size"]).agg(
        n_passengers=("PassengerId","count"),
        avg_fare=("Fare","mean"),
        min_fare=("Fare","min"),
        max_fare=("Fare","max")
    )

    return table.reset_index().sort_values(["Pclass","family_size"])


def last_names(df):

    last = df["Name"].str.split(",").str[0]
    return last.value_counts()


def visualize_families(table):

    fig = px.line(
        table,
        x="family_size",
        y="avg_fare",
        color="Pclass",
        markers=True
    )
    return fig
