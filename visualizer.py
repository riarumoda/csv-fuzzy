# visualizer.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def plot_risk_distribution(df):
    bins = [0, 33, 66, 100]
    labels = ['Low', 'Medium', 'High']
    df['risk_level'] = pd.cut(df['risk_score'], bins=bins, labels=labels)
    
    counts = df['risk_level'].value_counts().sort_index()

    counts.plot(kind='bar', color=['green', 'orange', 'red'])
    plt.title('Heart Disease Risk Levels')
    plt.xlabel('Risk Level')
    plt.ylabel('Number of People')
    plt.tight_layout()
    plt.show()

def plot_fuzzy_membership_functions():
    # Example: visualize fuzzy membership for 'age'
    age = ctrl.Antecedent(np.arange(20, 81, 1), 'age')
    age.automf(3)  # Generates 'poor', 'average', 'good'

    plt.figure()
    for label in age.terms:
        plt.plot(age.universe, age[label].mf, label=label)

    plt.title('Fuzzy Set: Age')
    plt.xlabel('Age')
    plt.ylabel('Membership')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
