from data_loader import load_heart_data
from fuzzy_logic import create_fuzzy_system, evaluate_risk
from visualizer import plot_risk_distribution, plot_fuzzy_membership_functions

def main():
    csv_path = 'heart.csv'  # Replace with your path
    df = load_heart_data(csv_path)

    risk_sim = create_fuzzy_system()
    df['risk_score'] = df.apply(lambda row: evaluate_risk(row, risk_sim), axis=1)

    plot_risk_distribution(df)
    plot_fuzzy_membership_functions()

if __name__ == '__main__':
    main()
