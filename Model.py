# Libraries importation
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import umap

# Displays for output
from colorama import Fore, Style
from IPython.display import clear_output
from tqdm import tqdm

# Sklearn:
import sklearn 
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
import optuna
import shap
import lime
from itertools import product
from tools import generate_param_grid, get_param_combinations, count_combinations, generate_random_numbers, random_number_dict

class Model:
     
    """
---------------   Initialisation and basic function   ---------------
    """

    def __init__(self, classifier):
        self.model = classifier
        self.random_state = 3
        self.params = {'random_state': self.random_state}

    def get_params(self):
        self.params['random_state'] = self.random_state
        return self.params

    def fit(self, X, y):
        self.params['random_state'] = self.random_state
        self.model.set_params(**self.params)
        self.model.fit(x, Y)
    
    def predict(self, X):
        return self.model.predict(X)    
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)
    

    """
---------------   Cross-validation score of the model   ---------------
    """


    def k_fold_cross_validation(self, X, y, k=5):
        """
        Performs k-fold cross-validation for a given model and dataset.

        Parameters:
            model: The machine learning model to evaluate.
            X (numpy.ndarray): The feature matrix.
            y (numpy.ndarray): The target vector.
            k (int): Number of folds for cross-validation.

        Returns:
            float: The average accuracy across all folds.
        """
        n = X.shape[0]
        fold_size = n // k
        scores = []

        for i in range(k):
            # Splitting data into training and validation sets
            validation_X = pd.DataFrame(X[i * fold_size: (i + 1) * fold_size], columns=X.columns)
            validation_y = y[i * fold_size: (i + 1) * fold_size]
            train_X = pd.DataFrame(np.concatenate([X[:i * fold_size], X[(i + 1) * fold_size:]]), columns=X.columns)
            train_y = np.concatenate([y[:i * fold_size], y[(i + 1) * fold_size:]])

            # Fitting the model
            self.fit(train_X, train_y)

            # Calculating accuracy
            score = self.score(validation_X, validation_y)
            scores.append(score)

        # Returning the average accuracy
        return sum(scores) / k
    

    """
---------------   Optimization of the model   ---------------
    """

    def optimize_with_optuna(self, x, Y, n_trials = 50):
        def objective(trial):
            # Define search space for hyperparameters
            params = {}
            if isinstance(self.model, GradientBoostingClassifier):
                params['learning_rate'] = trial.suggest_loguniform('learning_rate', 0.001, 0.1)
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 400)
                params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
                params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
                params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
                params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 20)
            elif isinstance(self.model, XGBClassifier) : 
                params = {
                    'objective': 'multi:softmax',
                    'n_estimators': trial.suggest_int('n_estimators', 50,500),
                    'num_class': 3,
                    'eval_metric': 'mlogloss',
                    'booster': trial.suggest_categorical('booster', ['gbtree']),
                    'max_depth': trial.suggest_int('max_depth', 3, 6),
                    'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 3),
                    'subsample': trial.suggest_float('subsample', 0.4, 0.9),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'learning_rate': trial.suggest_float('learning_rate', 0.007, 0.06),
                }
            elif isinstance(self.model, RandomForestClassifier):
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 400)
                params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
                params['max_samples'] = trial.suggest_float('max_samples', 0.1, 1.0)
                params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
                params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 20)
            elif isinstance(self.model, SVC):
                params['max_iter'] = 1000
                params['C'] = trial.suggest_int('C', 1e-5, 1e5)
                params['degree'] = trial.suggest_int('degree', 2, 5)
                params['gamma'] = trial.suggest_float('gamma', 1e-5, 1e3,log=True)
                params['kernel'] = trial.suggest_categorical('kernel', ['linear','poly'])

            # Initialize model with hyperparameters
            if isinstance(self.model, SVC):
                self.model = self.model.set_params(**params, random_state = self.random_state, probability = True)
            else : 
                self.model = self.model.set_params(**params, random_state = self.random_state)

            return self.k_fold_cross_validation(x, Y, k=5)
        
        # Create Optuna study object
        study = optuna.create_study(direction='maximize')

        # Run optimization
        study.optimize(objective, n_trials=n_trials)

        # Access best hyperparameters
        best_params = study.best_params
        self.model.set_params(**best_params)
        self.params = best_params
        
        # Displays of best_params in the os
        clear_output(wait=True)
        
        print(f'Best hyperparameters with optuna : {best_params}')
        
        return best_params
    
    def optimize(self, x, Y, num_samples = 3, n_trials = 50):
        # Define the gridsearch space
        optimal_params = self.optimize_with_optuna(x,Y, n_trials)
        param_grid = generate_param_grid(optimal_params, num_samples = num_samples)
                
        # Initialize the model
        best_score = 0
        best_params = None
        nb_comb = count_combinations(param_grid)
        progress_bar = tqdm(total=nb_comb, desc="Progress of GridSearchCV")

        for params in get_param_combinations(param_grid) : 
            # Create model instance with current hyperparameters
            self.model = self.model.set_params(**params, random_state = self.random_state) 

            if self.k_fold_cross_validation(x, Y, k=5)>best_score:
                best_score = self.k_fold_cross_validation(x, Y, k=5)
                best_params = params
                
            progress_bar.update(1)

        self.model.set_params(**best_params)
        self.params = best_params
        self.params['random_state'] = self.random_state
        progress_bar.close()

        # Displays of best_params in the os
        print(f'Best hyperparameters with optuna - GridSearch : {Fore.BLUE}{best_params}{Style.RESET_ALL} \nwith a score: {Fore.BLUE}{best_score}{Style.RESET_ALL}; and the scorer: {Fore.BLUE}{self.scorer}{Style.RESET_ALL}')

    """
---------------   Interpretability methods   ---------------
    """

    def get_interpretability_methods(self, x_train, x_test, Y_train, Y_test, feature = None, index = 0, plot = False):
        interpretability_methods = {
            'SHAP': self.get_shap_values(x_train, x_test, feature, plot),
            'LIME': self.get_lime_explanation(x_train, x_test, index),
            'PI': self.get_pi_values(x_test, Y_test, Y_train)
        }
        return interpretability_methods

    def get_shap_values(self, x_train, x_test, feature = None, plot = False):
        if isinstance(self.model, RandomForestClassifier) or isinstance(self.model, GradientBoostingClassifier) :
            # use Tree Explainer SHAP to explain test set predictions
            explainer = shap.Explainer(self.predict, x_train)
            shap_values = explainer.shap_values(x_test)
            
            if plot :
                # Display SHAP's summary plot
                shap.summary_plot(shap_values,x_test)
            
            # Display SHAP's dependance plot
            if feature != None : 
                shap.dependence_plot(feature, shap_values, x_test)

            return shap_values

    def permutation_importance_feature(self, x_test, Y_test, n_permutations=100):
        baseline_score = self.score(x_test, Y_test)
        feature_importance = {}
        
        for i in range(x_test.shape[1]):
            scores = []
            for j in range(n_permutations):
                x_permuted = x_test.copy()
                x_permuted.iloc[:, i] = np.random.permutation(x_permuted.iloc[:, i])
                score = self.score(x_permuted, Y_test)
                scores.append(score)
            feature_importance[x_test.columns[i]] = abs(baseline_score - np.mean(scores))
        
        return feature_importance


    def get_pi_values(self, x_test, Y_test, plot = False):
        if isinstance(self.model, RandomForestClassifier) or isinstance(self.model, GradientBoostingClassifier):
            feature_importances = self.permutation_importance_feature(x_test, Y_test)
            sorted_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1])}
            total_importance = sum(sorted_importances.values())
            importance_percent = {k: v / total_importance * 100 for k, v in sorted_importances.items()}

            if plot:
                # Display
                plt.figure(figsize=(10, 6))
                plt.barh(list(importance_percent.keys()), list(importance_percent.values()), color='blue')
                plt.xlabel('Importance (%)')
                plt.ylabel('Variable')
                plt.title('Variables importance')
                plt.show()

            return importance_percent
            
        elif isinstance(self.model, SVC):
            print(Fore.RED + "RVI is not available for SVM" + Style.RESET_ALL)
        

    def get_lime_explanation(self, x_train, x_test, sample_idx = [0]):        
        # Initialize LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data = x_train.values, feature_names = x_train.columns, class_names=['event_1','event_0'], mode = 'classification')

        for index in sample_idx:
        # Select instance to explain
            instance_idx = index

            # Explain prediction
            explanation = explainer.explain_instance(x_test.iloc[instance_idx], self.model.predict_proba)

            # Show explanation
            print(f'Lime explanation for index {index}')
            explanation.show_in_notebook()
            