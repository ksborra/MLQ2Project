import numpy as np
import pandas as pd
import time
from collections import Counter
from math import log
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import pearsonr
import shap
import matplotlib.pyplot as plt


try:
    from optimized_weights import OPTIMIZED_HYPERPARAMS, OPTIMIZED_WEIGHTS
    print("Successfully imported params")
except ImportError:
    OPTIMIZED_HYPERPARAMS = {
        'n_trees': 100,
        'max_depth': 10,
        'update_frequency': 10,
        'decay_factor': 0.9,
        'min_samples_split': 2
    }
    OPTIMIZED_WEIGHTS = {
        'mutual_info': 0.25,
        'shap': 0.25,
        'permutation': 0.25,
        'pearson': 0.15,
        'variance': 0.10
    }
    print("Failed to import params, reverting to default")

MAX_SAMPLES_FOR_SHAP = 500  # To handle large datasets
HIGH_DIMENSION_THRESHOLD = 50
DATA_FILE = 'nursery.csv'

class DataAnalyzer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.problem_type = self.detect_problem_type()
        self.feature_stats = self.calculate_feature_stats()
        
    def detect_problem_type(self):
        unique_labels = len(np.unique(self.y))
        if unique_labels / len(self.y) < 0.05 or isinstance(self.y[0], (int, np.integer)):
            return 'classification' if unique_labels < 10 else 'regression'
        return 'regression' if isinstance(self.y[0], float) else 'classification'
    
    def calculate_feature_stats(self):
        stats = {
            'n_samples': self.X.shape[0],
            'n_features': self.X.shape[1],
            'high_dim': self.X.shape[1] > HIGH_DIMENSION_THRESHOLD,
            'pearson_corrs': [abs(pearsonr(self.X[:, i], self.y)[0]) 
                                if np.issubdtype(self.X[:, i].dtype, np.number) else 0 
                                for i in range(self.X.shape[1])],
            'mutual_info': mutual_info_classif(self.X, self.y) 
                            if self.problem_type == 'classification' 
                            else mutual_info_regression(self.X, self.y)
        }
        stats['linearity_score'] = np.mean(stats['pearson_corrs'])
        return stats


class DecisionNode:
    def __init__(self, feature_idx=None, threshold=None, value=None, 
                 right=None, left=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.value = value
        self.right_branch = right
        self.left_branch = left

class ImprovedDecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, feature_weights=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_weights = feature_weights
        self.tree = None

    def _entropy(self, y):
        counts = Counter(y)
        entropy = 0.0
        for label in counts:
            prob = counts[label] / len(y)
            entropy -= prob * log(prob, 2) if prob > 0 else 0
        return entropy

    def _best_split(self, X, y):
        best_gain = -1
        best_idx, best_thresh = None, None

        if self.feature_weights is not None:
            features = np.random.choice(
                X.shape[1], 
                size=int(np.sqrt(X.shape[1])), 
                p=self.feature_weights/np.sum(self.feature_weights),
                replace=False
            )
        else:
            print("ERROR: feature weights is None")
            features = np.random.permutation(X.shape[1])[:int(np.sqrt(X.shape[1]))]

        for idx in features:
            thresholds = np.unique(X[:, idx])
            for thresh in thresholds:
                left_mask = X[:, idx] <= thresh
                if left_mask.sum() < self.min_samples_split or (~left_mask).sum() < self.min_samples_split:
                    continue
                    
                y_left, y_right = y[left_mask], y[~left_mask]
                gain = self._entropy(y) - (len(y_left)/len(y))*self._entropy(y_left) - (len(y_right)/len(y))*self._entropy(y_right)
                
                if gain > best_gain:
                    best_gain, best_idx, best_thresh = gain, idx, thresh
                    
        return best_idx, best_thresh

    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return DecisionNode(value=Counter(y).most_common(1)[0][0])
            
        idx, thresh = self._best_split(X, y)
        if idx is None:
            return DecisionNode(value=Counter(y).most_common(1)[0][0])
            
        left_mask = X[:, idx] <= thresh
        return DecisionNode(
            feature_idx=idx,
            threshold=thresh,
            right=self._build_tree(X[left_mask], y[left_mask], depth + 1),
            left=self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        )

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
        return self

    def predict(self, X):
        return [self._predict_single(x, self.tree) for x in X]

    def _predict_single(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.right_branch)
        return self._predict_single(x, node.left_branch)

class ImprovedRandomForest:
    def __init__(self, n_trees=100, max_depth=5, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.feature_weights = None
        self.analyzer = None
        self.train_time = 0
        self.accuracy_history = []


    def _normalize(self, arr): # normalizes to [0,1] range

        arr = np.nan_to_num(arr)  # Handle NaN/Inf
        if np.ptp(arr) == 0:  # All values identical
            return np.ones_like(arr) / len(arr)
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    def _calculate_feature_weights(self, X, y):
        self.analyzer = DataAnalyzer(X, y)
        active_weights = {k: v for k, v in self.metric_weights.items()}
        scores = {}
        
        # Auto-disable metrics based on dataset analysis
        if self.analyzer.feature_stats['high_dim']:
            print("Data has high dimension count; variance and shap will not be used")
        else:
            # Variance
            scores['variance'] = np.var(X, axis=0)

            # SHAP Values
            try:
                if self.analyzer.feature_stats['n_samples'] > MAX_SAMPLES_FOR_SHAP:
                    sample_idx = np.random.choice(X.shape[0], MAX_SAMPLES_FOR_SHAP, replace=False)
                    X_sample, y_sample = X[sample_idx], y[sample_idx]
                else:
                    X_sample, y_sample = X, y
                    
                if self.analyzer.problem_type == 'classification':
                    surrogate = RandomForestClassifier(n_estimators=3, max_depth=5).fit(X_sample, y_sample)
                else:
                    surrogate = RandomForestRegressor(n_estimators=3, max_depth=5).fit(X_sample, y_sample)
                    
                explainer = shap.TreeExplainer(surrogate)
                shap_values = explainer.shap_values(X_sample)
                
                if isinstance(shap_values, list):
                    # (n_classes, n_samples, n_features)
                    shap_array = np.stack(shap_values)
                    scores['shap'] = np.mean(np.abs(shap_array), axis=(0, 1))
                else:
                    # (n_samples, n_features)
                    scores['shap'] = np.abs(shap_values).mean(0)
                    
                scores['shap'] = scores['shap'][:X.shape[1]]
            except Exception as e:
                print(f"SHAP calculation failed: {str(e)}")
                active_weights.pop('shap', None)
                scores['shap'] = np.zeros(X.shape[1])

        if self.analyzer.feature_stats['linearity_score'] < 0.1:
            print("Data has low linearity score; pearson correlation will not be used")
        else:
            # Pearson Correlation
            scores['pearson'] = np.array(self.analyzer.feature_stats['pearson_corrs'])

        # Mutual Information
        scores['mutual_info'] = np.array(self.analyzer.feature_stats['mutual_info'])
        
        

        # Permutation Importance
        scores['permutation'] = np.zeros(X.shape[1])
        try:
            baseline = accuracy_score(y, self.predict(X))
            for i in range(X.shape[1]):
                X_perm = X.copy()
                X_perm[:, i] = np.random.permutation(X_perm[:, i])
                scores['permutation'][i] = baseline - accuracy_score(y, self.predict(X_perm))
        except:
            active_weights.pop('permutation', None)
        
        # this for loop is simply to fix bugs where certain variables are lists instead of np arrays and such.
        for metric in scores:
            if not isinstance(scores[metric], np.ndarray):
                scores[metric] = np.array(scores[metric])
                
            if scores[metric].ndim > 1:
                scores[metric] = scores[metric].mean(axis=0)
                
            if len(scores[metric]) != X.shape[1]:
                if len(scores[metric]) > X.shape[1]:
                    scores[metric] = scores[metric][:X.shape[1]]
                else:
                    scores[metric] = np.pad(scores[metric], (0, X.shape[1] - len(scores[metric])))

        total_weight = sum(active_weights.values())
        combined = np.zeros(X.shape[1])
        
        for metric, weight in active_weights.items():
            if metric not in scores:
                continue
                
            # Final validation check (might not be necessary anymore)
            if len(scores[metric]) != X.shape[1]:
                print(f"Skipping {metric} due to dimension mismatch")
                continue
                
            normalized = self._normalize(scores[metric])
            combined += normalized * (weight / total_weight)
            
        return combined / (combined.sum() + 1e-10)

    def fit(self, X, y, X_val=None, y_val=None):
        start_time = time.time()
        self.feature_weights = self._calculate_feature_weights(X, y)
        
        for i in range(self.n_trees):
            idx = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[idx], y[idx]
            
            tree = ImprovedDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                feature_weights=self.feature_weights
            ).fit(X_sample, y_sample)
            
            self.trees.append(tree)
            
            if X_val is not None and (i+1) % 10 == 0:
                self.accuracy_history.append(accuracy_score(y_val, self.predict(X_val)))
                
        self.train_time = time.time() - start_time
        return self

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return [Counter(col).most_common(1)[0][0] for col in preds.T]


class AdaptiveRandomForest(ImprovedRandomForest):
    def __init__(self, 
                 n_trees=OPTIMIZED_HYPERPARAMS['n_trees'],
                 max_depth=OPTIMIZED_HYPERPARAMS['max_depth'],
                 update_frequency=OPTIMIZED_HYPERPARAMS['update_frequency'],
                 decay_factor=OPTIMIZED_HYPERPARAMS['decay_factor'],
                 min_samples_split=OPTIMIZED_HYPERPARAMS['min_samples_split'],
                 metric_weights=OPTIMIZED_WEIGHTS):
        
        super().__init__(
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
        self.metric_weights = metric_weights
        self.update_frequency = update_frequency
        self.decay_factor = decay_factor
        self.feature_importances_ = None
        
    def _calculate_tree_importance(self, tree, n_features):
        importance = np.zeros(n_features)
        
        def _traverse(node):
            if node.feature_idx is not None:
                importance[node.feature_idx] += 1
                _traverse(node.right_branch)
                _traverse(node.left_branch)
                
        _traverse(tree.tree)
        return importance / importance.sum()
    
    def fit(self, X, y, X_val=None, y_val=None):
        start_time = time.time()
        n_features = X.shape[1]
        self.feature_importances_ = np.ones(n_features) / n_features
        self.accuracy_history = []
        
        for i in range(self.n_trees):
            if i % self.update_frequency == 0 and i > 0:
                recent_importances = np.stack([
                    self._calculate_tree_importance(tree, n_features)
                    for tree in self.trees[-self.update_frequency:]
                ])
                
                new_importances = recent_importances.mean(axis=0)
                self.feature_importances_ = (
                    self.decay_factor * self.feature_importances_ +
                    (1 - self.decay_factor) * new_importances
                )
                self.feature_importances_ /= self.feature_importances_.sum()
                
            if X_val is not None and (i+1) % 10 == 0:
                preds = self.predict(X_val)
                self.accuracy_history.append(accuracy_score(y_val, preds))
                
            tree = ImprovedDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                feature_weights=self.feature_importances_
            )
            idx = np.random.choice(len(X), len(X), replace=True)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
        
        self.train_time = time.time() - start_time
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


def load_data(filepath):
    df = pd.read_csv(filepath)
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def print_results(model_name, y_true, y_pred, train_time, acc_history, only_accuracy_and_time=False):
    print(f"\n{model_name} Results:")
    if only_accuracy_and_time:
        print(f"Accuracy: {(accuracy_score(y_true, y_pred) * 100):.2f}%")
        print(f"Training Time: {train_time:.2f}s")
    else:
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print(f"Accuracy: {(accuracy_score(y_true, y_pred) * 100):.2f}%")
        print(f"Training Time: {train_time:.2f}s")
        print(f"Average Convergence Time: {sum(acc_history) / len(acc_history)}")
        # print(np.array(acc_history))
        # print(*acc_history)
        plt.figure(figsize=(8, 5))
        plt.plot(range(10, 10 * (len(acc_history) + 1), 10), acc_history, marker='o', linestyle='-', color='b')
        plt.title(f"{model_name} Convergence History")
        plt.xlabel("Number of Trees")
        plt.ylabel("Convergence Time (s)")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    X, y = load_data(DATA_FILE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    # Sklearn Baseline
    start = time.time()
    sklearn_rf = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=5)
    sklearn_rf.fit(X_train, y_train)
    sklearn_time = time.time() - start
    sklearn_acc = [accuracy_score(y_test, 
                                  RandomForestClassifier(n_estimators=n, criterion='entropy', max_depth=5).fit(X_train, y_train).predict(X_test)) 
                                  for n in range(10, 101, 10)]

    # Improved RF
    improved_rf = AdaptiveRandomForest(n_trees=100, update_frequency=5, decay_factor=0.85)
    improved_rf.fit(X_train, y_train, X_test, y_test)

    print_results("Baseline", y_test, sklearn_rf.predict(X_test), 
                 sklearn_time, sklearn_acc, only_accuracy_and_time=False)
    print_results("Improved RF", y_test, improved_rf.predict(X_test),
                 improved_rf.train_time, improved_rf.accuracy_history, only_accuracy_and_time=False)