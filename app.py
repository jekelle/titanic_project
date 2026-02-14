import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("titanic.csv")


def survival_demographics(df):

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

    summary = summary.sort_values(
        by=["Pclass", "Sex", "AgeGroup"]
    )

    return summary


summary = survival_demographics(df)

print("\nSurvival Demographics Summary:\n")
print(summary)


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


print("\nQuestion:")
print("Did women in first class have higher survival rates than men in other classes?")

fig = visualize_demographic(summary)
st.plotly_chart(fig)




