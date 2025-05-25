# fuzzy_logic.py
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def create_fuzzy_system():
    age = ctrl.Antecedent(np.arange(20, 81, 1), 'age')
    cp = ctrl.Antecedent(np.arange(0, 4, 1), 'cp')
    trestbps = ctrl.Antecedent(np.arange(90, 201, 1), 'trestbps')
    fbs = ctrl.Antecedent(np.arange(0, 2, 1), 'fbs')
    thalach = ctrl.Antecedent(np.arange(60, 201, 1), 'thalach')
    thal = ctrl.Antecedent(np.arange(0, 3, 1), 'thal')
    risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

    age.automf(3)
    cp.automf(3)
    trestbps.automf(3)
    fbs.automf(3)
    thalach.automf(3)
    thal.automf(3)

    risk['low'] = fuzz.trimf(risk.universe, [0, 0, 50])
    risk['medium'] = fuzz.trimf(risk.universe, [25, 50, 75])
    risk['high'] = fuzz.trimf(risk.universe, [50, 100, 100])

    rules = [
    ctrl.Rule(age['poor'] | cp['poor'] | trestbps['poor'] | thalach['poor'] | thal['poor'] | fbs['poor'], risk['high']),
    ctrl.Rule(age['average'] & cp['average'] & trestbps['average'] & fbs['average'], risk['medium']),
    ctrl.Rule(age['good'] & cp['good'] & trestbps['good'] & thalach['good'] & fbs['good'], risk['low']),
    # Optional broad rule instead of invalid fallback
    ctrl.Rule(age['average'] | cp['average'] | trestbps['average'], risk['medium'])
]

    risk_ctrl = ctrl.ControlSystem(rules)
    return risk_ctrl

def evaluate_risk(row, fuzzy_system):
    sim = ctrl.ControlSystemSimulation(fuzzy_system)
    try:
        sim.input['age'] = row['age']
        sim.input['cp'] = row['cp']
        sim.input['trestbps'] = row['trestbps']
        sim.input['fbs'] = row['fbs']
        sim.input['thalach'] = row['thalach']
        sim.input['thal'] = row['thal']
        sim.compute()
        return sim.output['risk']
    except Exception as e:
        print(f"Failed to compute risk for row {row.to_dict()} - Error: {e}")
        return np.nan
