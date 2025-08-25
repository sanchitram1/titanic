import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('train.csv')

def last_minute_mask(df, q=0.10):
    q_low = df["Fare"].quantile(q)
    q_high = df["Fare"].quantile(1 - q)
    return (df["Embarked"] == "Q") & ((df["Fare"] <= q_low) | (df["Fare"] >= q_high))

def summarize_subset(sub):
    total = len(sub)
    survived = int(sub["Survived"].sum())
    not_survived = total - survived
    survival_rate = 0.0 if total == 0 else survived / total * 100

    male = (sub["Sex"] == "male").sum()
    female = (sub["Sex"] == "female").sum()
    male_rate = 0.0 if male == 0 else sub.loc[sub["Sex"]=="male","Survived"].mean()*100
    female_rate = 0.0 if female == 0 else sub.loc[sub["Sex"]=="female","Survived"].mean()*100

    print("Survival Analysis")
    print(f"Total Passengers: {total}")
    print(f"Survived: {survived}")
    print(f"Did Not Survive: {not_survived}")
    print(f"Survival Rate: {survival_rate:.2f}%")
    print("-"*10)
    print("Gender Distribution")
    print(f"Male Passengers: {male}")
    print(f"Female Passengers: {female}")
    print(f"Male Survival Rate: {male_rate:.2f}%")
    print(f"Female Survival Rate: {female_rate:.2f}%")
    print("-"*10)
    print("Age Distribution")
    age_stats = sub["Age"].dropna().describe()
    print("Age Summary Statistics:")
    print(age_stats)
    print(f"\nNumber of missing age values: {sub['Age'].isna().sum()}")
    print("-"*10)

def plot_age_fare(sub):
    sub2 = sub.dropna(subset=["Age","Fare"])
    surv0 = sub2[sub2["Survived"]==0]
    surv1 = sub2[sub2["Survived"]==1]
    plt.figure()
    ax = plt.gca()
    ax.scatter(surv0["Age"] + np.random.uniform(-0.3, 0.3, len(surv0)),
           surv0["Fare"] + np.random.uniform(-0.5, 0.5, len(surv0)),
           alpha=0.3, label="No")

    ax.scatter(surv1["Age"] + np.random.uniform(-0.3, 0.3, len(surv1)),
           surv1["Fare"] + np.random.uniform(-0.5, 0.5, len(surv1)),
           alpha=0.4, label="Yes")

    ax.set_title('Survival Rate of "Last Minute Ticket" by Age and Fare')
    ax.set_xlabel("Age")
    ax.set_ylabel("Fare")
    ax.legend(title="Survived")
    plt.show()

summarize_subset(df)
plot_age_fare(df)