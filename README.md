# Titanic Survival Analysis

This project analyzes the Titanic dataset from Kaggle to explore patterns in passenger survival using Python and Streamlit.

## Exercises Covered

### **Exercise 1: Survival Patterns**
- Created age groups (Child, Teen, Adult, Senior)
- Grouped data by class, sex, and age group
- Calculated survival counts and rates
- Visualized using Plotly

**Key Finding:**  
Women in first class had significantly higher survival rates than men in all classes.

### **Exercise 2: Family Size & Wealth**
- Created a `family_size` feature
- Grouped by class and family size
- Computed statistics on ticket fare
- Extracted last names and compared frequency

**Key Finding:**  
Larger families in higher classes tended to pay higher average fares. Last name frequency aligned with larger family groupings.

## Usage

Install dependencies:

```bash
pip3 install pandas plotly streamlit
