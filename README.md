# Assignment-1-apr-
assignment 1 for simple linear regression
## Dataset
The dataset used is `Salary_Data.csv
The dataset used is `Salary_Data.csv`, which contains two columns:

| Column Name        | Description                         |
|-------------------|-------------------------------------|
| YearsExperience    | Number of years of experience       |
| Salary             | Corresponding salary (target value) |

## Steps
1. Load the dataset with pandas.
2. Split into training (80%) and test (20%) sets.
3. Train `LinearRegression` model.
4. Predict salaries on the test set.
5. Evaluate with **Mean Squared Error** and **R² Score**.
6. Visualize data and regression line.

## How to run
1. Install dependencies:
```bash
pip install pandas numpy matplotlib scikit-learn
