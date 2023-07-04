import joblib
study = joblib.load("results/hyperpar_search_results/studies/all_2022_test_std_scaler.pkl")
print("Best trial until now:")
print(" Value: ", study.best_trial.value)
print(" Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")