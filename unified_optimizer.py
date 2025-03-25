import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from custom_rf import AdaptiveRandomForest, load_data
from sklearn.model_selection import train_test_split

def optimize():
    X, y = load_data('nursery.csv')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    def objective(trial: optuna.Trial):
        params = {
            # Core parameters
            'n_trees': trial.suggest_int('n_trees', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            
            # Adaptive parameters
            'update_frequency': trial.suggest_int('update_frequency', 5, 50),
            'decay_factor': trial.suggest_float('decay_factor', 0.1, 0.99),
            
            # Metric weights (automatically normalized)
            'mutual_info': trial.suggest_float('mutual_info', 0.0, 1.0),
            'shap': trial.suggest_float('shap', 0.0, 1.0),
            'permutation': trial.suggest_float('permutation', 0.0, 1.0),
            'pearson': trial.suggest_float('pearson', 0.0, 1.0),
            'variance': trial.suggest_float('variance', 0.0, 1.0)
        }
        

        # Normalize weights
        weights_total = sum([params[k] for k in ['mutual_info', 'shap', 'permutation', 'pearson', 'variance']])
        metric_weights = {k: params.pop(k)/weights_total for k in ['mutual_info', 'shap', 'permutation', 'pearson', 'variance']}
        
        params['n_trees'] = min(50, params['n_trees'])

        # Efficient early evaluation
        model = AdaptiveRandomForest(
            **params,
            metric_weights=metric_weights,
            # n_trees=min(50, params['n_trees'])  # Evaluate with fewer trees during optimization
        )
        model.fit(X_train, y_train, X_val, y_val)
        
        # Intermediate value for pruning
        if len(model.accuracy_history) > 0:
            trial.report(model.accuracy_history[-1], step=len(model.accuracy_history))
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        return 1 - model.accuracy_history[-1] if len(model.accuracy_history) > 0 else 0.5

    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(n_startup_trials=20),
        pruner=HyperbandPruner(min_resource=1, max_resource=50, reduction_factor=3)
    )
    
    study.optimize(objective, n_trials=200, n_jobs=-1, show_progress_bar=True)
    
    # Save best parameters
    best_params = study.best_params.copy()
    
    # Separate hyperparameters and weights
    optimized_hyperparams = {
        'n_trees': best_params['n_trees'],
        'max_depth': best_params['max_depth'],
        'update_frequency': best_params['update_frequency'],
        'decay_factor': best_params['decay_factor'],
        'min_samples_split': best_params['min_samples_split']
    }
    
    weights_total = sum([best_params[k] for k in ['mutual_info', 'shap', 'permutation', 'pearson', 'variance']])
    optimized_weights = {
        k: best_params[k]/weights_total
        for k in ['mutual_info', 'shap', 'permutation', 'pearson', 'variance']
    }
    
    # Write to file
    with open("optimized_weights.py", "w") as f:
        f.write("OPTIMIZED_HYPERPARAMS = " + repr(optimized_hyperparams) + "\n\n")
        f.write("OPTIMIZED_WEIGHTS = " + repr(optimized_weights) + "\n")
    
    return optimized_hyperparams, optimized_weights

if __name__ == "__main__":
    hyperparams, weights = optimize()
    print("Optimized Hyperparameters:", hyperparams)
    print("Optimized Weights:", weights)