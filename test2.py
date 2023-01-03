import optuna
study_name="example-study"
study = optuna.load_study(study_name, storage=study_name)
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
print(df)