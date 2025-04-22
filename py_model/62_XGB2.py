import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import shap
from sklearn.calibration import CalibratedClassifierCV
import pickle
import json
import os
from scipy import stats
import logging
from datetime import datetime
from tqdm import tqdm
import warnings

# 过滤XGBoost的特定警告
warnings.filterwarnings('ignore', message='.*use_label_encoder.*')

def setup_logger(model_name):
    """Set up logger for the model."""
    log_dir = '/Users/maguoli/Documents/Development/Predictive/Models'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f'{model_name}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),  # 使用追加模式
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def load_and_preprocess_data():
    """加载和预处理数据，并进行训练集和测试集的分层划分"""
    # 读取数据
    df = pd.read_csv('/Users/maguoli/Documents/Development/Predictive/data/16_ML.csv')
    
    # 定义特征
    weights = df['WTDRD1']
    numeric_features = ['Age', 'BMI']
    categorical_features = ['Gender', 'Education', 'Marriage', 'Smoke', 
                          'Alcohol', 'Employment', 'ActivityLevel']
    
    # 准备X和y
    X = pd.concat([
        df[['DII_food']],
        df[numeric_features],
        df[categorical_features]
    ], axis=1)
    
    y = df['Epilepsy']
    
    # 使用分层抽样划分训练集和测试集（8:2）
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    return (X_train, X_test, y_train, y_test, 
            weights_train, weights_test,
            numeric_features, categorical_features)

def evaluate_dii_shap_separation(model, X, preprocessor):
    """评估DII_food的SHAP值分离程度"""
    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    X_transformed = preprocessor.transform(X)
    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # 获取DII_food的SHAP值（第一列）
    dii_shap = shap_values[:, 0]
    
    # 获取DII_food的原始值
    dii_values = X['DII_food'].values
    
    # 计算DII高值（>75%分位数）和低值（<25%分位数）的SHAP值差异
    high_dii_mask = dii_values > np.percentile(dii_values, 75)
    low_dii_mask = dii_values < np.percentile(dii_values, 25)
    
    high_dii_shap = dii_shap[high_dii_mask]
    low_dii_shap = dii_shap[low_dii_mask]
    
    # 计算分离度指标
    # 1. 两组SHAP值的中位数差异
    median_diff = np.median(high_dii_shap) - np.median(low_dii_shap)
    
    # 2. 两组SHAP值的重叠程度（使用KS检验的统计量）
    ks_stat, _ = stats.ks_2samp(high_dii_shap, low_dii_shap)
    
    return median_diff, ks_stat

def random_search_cv(X_train, X_test, y_train, y_test, weights_train, weights_test, numeric_features, categorical_features, n_trials=30, logger=None):
    """带有SHAP约束的随机搜索交叉验证"""
    if logger is None:
        logger = logging.getLogger()
    
    logger.info("Starting random search with cross-validation")
    logger.info(f"Number of trials: {n_trials}")
    
    # 创建预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['DII_food'] + numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])
    
    # 定义参数范围
    param_ranges = {
        'max_depth': (3, 15),           # 树的最大深度
        'min_child_weight': (1, 10),    # 子节点中所需的最小样本权重和
        'gamma': (0, 5),                # 分裂所需的最小损失减少
        'subsample': (0.5, 1.0),        # 训练每棵树时使用的样本比例
        'colsample_bytree': (0.5, 1.0), # 训练每棵树时使用的特征比例
        'learning_rate': (0.01, 0.1),   # 学习率
        'n_estimators': (100, 1000)     # 树的数量
    }
    
    logger.info("Parameter ranges:")
    for param, range_val in param_ranges.items():
        logger.info(f"{param}: {range_val}")
    
    best_score = float('-inf')
    best_params = None
    best_model = None
    
    # 定义交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for trial in tqdm(range(n_trials), desc="Hyperparameter optimization"):
        # 随机生成参数
        params = {
            'max_depth': np.random.randint(param_ranges['max_depth'][0], param_ranges['max_depth'][1]),
            'min_child_weight': np.random.randint(param_ranges['min_child_weight'][0], param_ranges['min_child_weight'][1]),
            'gamma': np.random.uniform(param_ranges['gamma'][0], param_ranges['gamma'][1]),
            'subsample': np.random.uniform(param_ranges['subsample'][0], param_ranges['subsample'][1]),
            'colsample_bytree': np.random.uniform(param_ranges['colsample_bytree'][0], param_ranges['colsample_bytree'][1]),
            'learning_rate': np.random.uniform(param_ranges['learning_rate'][0], param_ranges['learning_rate'][1]),
            'n_estimators': np.random.randint(param_ranges['n_estimators'][0], param_ranges['n_estimators'][1])
        }
        
        try:
            # 创建模型
            model = XGBClassifier(
                **params,
                objective='binary:logistic',
                eval_metric='auc',
                random_state=42
            )
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # 进行交叉验证
            scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
                w_train_fold, w_val_fold = weights_train.iloc[train_idx], weights_train.iloc[val_idx]
                
                pipeline.fit(X_train_fold, y_train_fold, classifier__sample_weight=w_train_fold)
                y_pred = pipeline.predict_proba(X_val_fold)[:, 1]
                
                # 计算加权AUC
                score = weighted_auc(y_val_fold, y_pred, w_val_fold)
                scores.append(score)
            
            mean_score = np.mean(scores)
            
            # 检查是否是新的最佳分数
            if mean_score > best_score:
                logger.info(f"\nTrial {trial + 1}: New best CV AUC found: {mean_score:.4f}")
                logger.info("Parameters:")
                for param, value in params.items():
                    logger.info(f"{param}: {value}")
                
                # 如果其他硬约束都满足，再检查SHAP分离度
                pipeline.fit(X_train, y_train, classifier__sample_weight=weights_train)
                median_diff, ks_stat = evaluate_dii_shap_separation(
                    pipeline.named_steps['classifier'],
                    X_train,
                    pipeline.named_steps['preprocessor']
                )
                
                # 设定SHAP分离度的硬约束阈值
                MIN_MEDIAN_DIFF = 0.2  # SHAP值中位数差异最小阈值
                MIN_KS_STAT = 0.3     # KS统计量最小阈值
                
                logger.info(f"SHAP metrics:")
                logger.info(f"Median difference: {median_diff:.4f}")
                logger.info(f"KS statistic: {ks_stat:.4f}")
                
                # 评估测试集性能
                y_pred_test = pipeline.predict_proba(X_test)[:, 1]
                test_score = weighted_auc(y_test, y_pred_test, weights_test)
                logger.info(f"Test set AUC: {test_score:.4f}")
                
                if median_diff >= MIN_MEDIAN_DIFF and ks_stat >= MIN_KS_STAT:
                    best_score = mean_score
                    best_params = params.copy()
                    best_model = pipeline
                    logger.info("Model satisfies all constraints - updating best model")
                else:
                    logger.info("Model does not satisfy SHAP constraints - skipping")
            
        except Exception as e:
            logger.error(f"Error in trial {trial + 1}: {str(e)}")
            continue
    
    if best_model is not None:
        logger.info("\nBest model found:")
        logger.info(f"Best CV AUC: {best_score:.4f}")
        logger.info("Best parameters:")
        for param, value in best_params.items():
            logger.info(f"{param}: {value}")
    else:
        logger.warning("No model found that satisfies all constraints!")
    
    return best_model, best_params, best_score

def weighted_auc(y_true, y_pred, weights):
    """计算加权AUC"""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred, sample_weight=weights)

def main():
    # 设置日志
    logger = setup_logger("62_XGB")
    logger.info("\n" + "="*50)  # 添加分隔符
    logger.info(f"Starting new training session at {datetime.now()}")
    
    try:
        # 加载和预处理数据
        logger.info("Loading and preprocessing data...")
        (X_train, X_test, y_train, y_test, 
         weights_train, weights_test,
         numeric_features, categorical_features) = load_and_preprocess_data()
        logger.info(f"Data loaded: {len(X_train)} training samples, {len(X_test)} testing samples")
        
        # 运行随机搜索
        best_model, best_params, best_score = random_search_cv(
            X_train, X_test, y_train, y_test, weights_train, weights_test,
            numeric_features, categorical_features,
            logger=logger
        )
        
        # 使用最佳参数训练最终模型
        if best_model is not None:
            logger.info("Training final model with best parameters...")
            
            # 保存最佳参数到JSON文件
            param_file = os.path.join('Models', '62_XGB_param.json')
            param_info = {
                'best_params': best_params,
                'best_score': best_score,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(param_file, 'w') as f:
                json.dump(param_info, f, indent=4)
            logger.info(f"Best parameters saved to: {param_file}")
            
            # 保存原始的XGBoost模型（用于SHAP分析）
            xgb_model_path = os.path.join('Models', '62_XGB_model_original.pkl')
            with open(xgb_model_path, 'wb') as f:
                pickle.dump(best_model, f)
            logger.info(f"Original XGBoost model saved to: {xgb_model_path}")
            
            # 校准模型概率
            calibrated_model = CalibratedClassifierCV(
                best_model.named_steps['classifier'],
                cv=5,
                method='sigmoid'
            )
            
            # 创建新的pipeline，包含校准的模型
            final_pipeline = Pipeline([
                ('preprocessor', best_model.named_steps['preprocessor']),
                ('classifier', calibrated_model)
            ])
            
            # 训练最终模型
            final_pipeline.fit(X_train, y_train)
            
            # 保存校准后的模型
            calibrated_model_path = os.path.join('Models', '62_XGB_model.pkl')
            with open(calibrated_model_path, 'wb') as f:
                pickle.dump(final_pipeline, f)
            
            logger.info(f"Calibrated model saved to: {calibrated_model_path}")
            
            # 在测试集上评估模型性能
            y_pred_test = final_pipeline.predict_proba(X_test)[:, 1]
            test_score = weighted_auc(y_test, y_pred_test, weights_test)
            logger.info(f"\nFinal model performance on test set:")
            logger.info(f"Weighted AUC: {test_score:.4f}")
            
            # 更新参数文件，添加测试集性能
            param_info['test_score'] = test_score
            with open(param_file, 'w') as f:
                json.dump(param_info, f, indent=4)
            
            logger.info(f"Final model and results saved to: {calibrated_model_path}")
            logger.info("Training completed successfully!")
        else:
            logger.warning("No model found that satisfies all constraints!")
            
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
