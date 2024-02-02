import cudf
import numpy as np
import pandas as pd
from cuml.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
import os
from cuml.svm import LinearSVR

file_paths = {
    "age_thinned": "",
    "mri_code_thinned": "",
    "data_thinned": ""
}  

# Loading data using cuDF
y_thinned = cudf.read_parquet(file_paths["age_thinned"])
mri_code_thinned = pd.read_parquet(file_paths["mri_code_thinned"])
X_thinned = pd.read_parquet(file_paths["data_thinned"])
X = cudf.from_pandas(X_thinned)

mri_code_thinned['group'] = mri_code_thinned['mri_code'].str.split('.').str[0]
group_dict = mri_code_thinned.groupby('group').apply(lambda df: df.index.tolist()).to_dict()
groups = mri_code_thinned['group']


  
def generate_group_splits(groups, n_splits, seed):
    unique_groups = np.unique(groups)
    np.random.seed(seed)
    np.random.shuffle(unique_groups)
    split_size = len(unique_groups) // n_splits
    group_splits = [unique_groups[i:i + split_size] for i in range(0, len(unique_groups), split_size)]
    return group_splits

def grid_search(X_full, y_full, model, param_grid, n_runs, n_splits, groups, mri_codes):
    results = []
    folder_name_components = f"Name of model"
    os.makedirs(folder_name_components, exist_ok=True)
    for C_value in param_grid['C']:
        folder_name = f"{folder_name_components}/{C_value}_Results"
        os.makedirs(folder_name, exist_ok=True)

        model.set_params(C=C_value)
        overall_metrics = {'C': C_value, 'R2': [], 'MAE': [], 'RMSE': [], 'PearsonR': [], 'Avg_Training_Time': []}
        folder_name = f"{folder_name_components}/{C_value}_results"


        for run in range(n_runs):
            group_splits = generate_group_splits(groups, n_splits, seed=run)  
            run_metrics = {'R2': [], 'MAE': [], 'RMSE': [], 'PearsonR': []}
            run_training_times = []
            run_training_times_forsaving = []
            run_predictions = []

            for fold, test_groups in enumerate(group_splits):
                test_indices = groups.isin(test_groups)
                train_indices = ~test_indices

                X_train, X_test = X_full[train_indices], X_full[test_indices]
                y_train, y_test = y_full[train_indices], y_full[test_indices]
                mri_codes_test = mri_codes[test_indices]['mri_code']

                start_time = time.time()
                model.fit(X_train, y_train)
                elapsed_time = time.time() - start_time
                run_training_times.append(elapsed_time)
                run_training_times_forsaving.append({'Run': run, 'Fold': fold, 'TrainingTime': elapsed_time})
                preds = model.predict(X_test)
                preds_print = preds.tolist()

                results_fold = pd.DataFrame({
                    'MRI_Code': mri_codes_test.values,  # Ensuring it's a list of values
                    'Predictions': preds_print
                })

                run_predictions.append(results_fold)
                y_test_series = y_test[y_test.columns[0]]
                run_metrics['R2'].append(r2_score(y_test_series, preds))
                run_metrics['MAE'].append(mean_absolute_error(y_test_series, preds))
                run_metrics['RMSE'].append(mean_squared_error(y_test_series, preds, squared=False))
                pearson_r = np.corrcoef(y_test_series.to_numpy(), preds.to_numpy())[0, 1]
                
                run_metrics['PearsonR'].append(pearson_r)

            run_training_times_forsaving_df = pd.DataFrame(run_training_times)
            training_times_filename = f"{folder_name}/C={C_value}_run={run}_training_times.csv"
            run_training_times_forsaving_df.to_csv(training_times_filename, index=False)
            
            filename = f'{folder_name}/C={C_value}_run={run}_predictions.csv'
            run_predictions_df = pd.concat(run_predictions, ignore_index=True)
            run_predictions_df.to_csv(filename, index=False)

            for metric in overall_metrics:
                if metric != 'C':
                    if metric != 'Avg_Training_Time':
                        overall_metrics[metric].append(sum(run_metrics[metric]) / len(run_metrics[metric]))

            overall_metrics['Avg_Training_Time'].append(sum(run_training_times) / len(run_training_times))

        avg_metrics = {metric: sum(values) / len(values) for metric, values in overall_metrics.items() if metric != 'C'}
        results.append({**avg_metrics, 'C': C_value})

    results_df = pd.DataFrame(results)
    print("Grid search completed.")

    # Save grid search results to file
    results_filename = f"{folder_name_components}/grid_search_results_CUDA_thinned.csv"
    results_df.to_csv(results_filename, index=False)



model = LinearSVR(max_iter=10000, epsilon=0)



C_values = [0.001]
param_grid = {'C': C_values}

grid_search(X, y_thinned, model, param_grid, n_runs=1, n_splits=10, groups=groups, mri_codes=mri_code_thinned)