
import pandas as pd
import plotly.express as px
import streamlit as st



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

    summary["survival_rate"] = (
        summary["n_survivors"] / summary["n_passengers"]
    )

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

    name_counts = df["last_name"].value_counts()

    return name_counts


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



st.title("Titanic Data Analysis")

df = pd.read_csv("titanic.csv")



st.header("Exercise 1: Survival Patterns")

summary = survival_demographics(df)

st.subheader("Survival Demographics Table")
st.dataframe(summary)

st.write(
    "Did women in first class have higher survival rates than men in other classes?"
)

fig1 = visualize_demographic(summary)
st.plotly_chart(fig1)

st.write(
    "Women in first class had significantly higher survival rates "
    "than men across all classes."
)



st.header("Exercise 2: Family Size and Wealth")

family_summary = family_groups(df)

st.subheader("Family Size Summary")
st.dataframe(family_summary)

name_counts = last_names(df)

st.subheader("Last Name Counts (Top 20)")
st.write(name_counts.head(20))

st.write(
    "Do larger families in higher passenger classes tend to pay higher average fares?"
)

fig2 = visualize_families(family_summary)
st.plotly_chart(fig2)

st.write(
    "Larger families in first class generally paid higher fares, "
    "while larger families in third class paid significantly lower fares. "
    "Repeated last names correspond to larger family groups, "
    "which aligns with the family size analysis."
)

