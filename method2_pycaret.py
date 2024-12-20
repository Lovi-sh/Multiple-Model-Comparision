import pandas as pd
from pycaret.regression import *

data = pd.read_csv('Fuel_cell_performance_data-Full.csv')

s = setup(
    data=data,
    target='Target4',
    train_size=0.7,
)

print("\nComparing All Models:")
best = compare_models()

print("\nCreating CatBoost Model:")
catboost_model = create_model('catboost')
