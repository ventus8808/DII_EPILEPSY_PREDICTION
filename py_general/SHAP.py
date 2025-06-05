import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
from catboost import CatBoostClassifier
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import os
import yaml
import random
from matplotlib import cm

def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_data(config):
    """Load and preprocess data"""
    df = pd.read_csv(config['data_path'])
    
    weights = df['WTDRD1'] if 'WTDRD1' in df.columns else None
    covariables = config['covariates']
    
    # Keep consistent with original CatBoost model feature processing
    X = pd.concat([
        df[config['exposure']],
        df[covariables]
    ], axis=1)
    y = df[config['outcome']]
    
    # Calculate DII quartiles, only for stratified sampling
    df['DII_Q'] = pd.qcut(df[config['exposure']], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    return X, y, weights, df['DII_Q']

def load_model():
    """Load trained CatBoost model"""
    model_path = 'model/SHAP.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def create_shap_explainer(model, X):
    """Create SHAP explainer"""
    # For CatBoost model, we use TreeExplainer
    explainer = shap.TreeExplainer(model)
    return explainer

def calculate_shap_values(explainer, X):
    """Calculate SHAP values"""
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
    
    # For binary classification, CatBoost's shap_values may return two arrays (one for each class)
    # We only need the SHAP values for the positive class (class 1)
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values = shap_values[1]  # Get SHAP values for positive class
        
    return shap_values

def get_sample_indices(y, dii_q):
    """Get stratified sample indices based on DII quartiles"""
    # Get case and control indices
    case_indices = np.where(y == 1)[0]
    control_indices = np.where(y == 0)[0]
    
    # Set sample sizes for each quartile
    q_sample_sizes = {
        'Q1': 2000,
        'Q2': 1500,
        'Q3': 500,
        'Q4': 500
    }
    
    # Sample from each quartile
    control_samples = []
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        # Get control indices for current quartile
        q_control_indices = control_indices[dii_q[control_indices] == q]
        # Determine actual sample size (not exceeding available samples)
        n_samples = min(q_sample_sizes[q], len(q_control_indices))
        # Random sampling
        sampled_indices = np.random.choice(q_control_indices, size=n_samples, replace=False)
        control_samples.append(sampled_indices)
    
    # Combine all sampled controls and all cases
    summary_indices = np.concatenate([case_indices] + control_samples)
    
    return summary_indices

def rename_features(X, config):
    """Rename features for display"""
    # Create a copy to avoid modifying the original dataframe
    X_display = X.copy()
    
    # Define feature name mapping
    feature_mapping = {
        'DII_food': 'DII',
        'ActivityLevel': 'PA'
    }
    
    # Rename columns
    X_display.columns = [feature_mapping.get(col, col) for col in X_display.columns]
    
    return X_display

def plot_shap_beeswarm(shap_values, X, output_dir, config):
    """Plot SHAP beeswarm (color version)"""
    # Rename features for display
    X_display = rename_features(X, config)
    
    # Set figure size and DPI - use a more square aspect ratio
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Plot beeswarm
    shap.summary_plot(
        shap_values, 
        X_display,
        plot_type="dot",
        show=False,
        sort=True,
        max_display=10
    )
    
    # Customize plot
    ax = plt.gca()
    
    # Set title and labels
    plt.title('SHAP Summary Plot', fontsize=16, fontweight='bold')
    plt.xlabel('SHAP value (impact on model output)', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'SHAP_beeswarm.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Color beeswarm plot saved to: {output_path}")
    
    # Close figure
    plt.close()

def plot_shap_beeswarm_bw(shap_values, X, output_dir, config):
    """Plot SHAP beeswarm (black and white version)"""
    # Rename features for display
    X_display = rename_features(X, config)
    
    # Set figure size and DPI - use a more square aspect ratio
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Get current axis
    ax = plt.gca()
    
    # Set gray background
    ax.set_facecolor('#f0f0f0')  # Light gray background
    plt.gcf().set_facecolor('white')  # Keep overall background white
    
    # Remove grid lines since background is already gray
    ax.grid(False)
    
    # Plot black and white beeswarm
    shap.summary_plot(
        shap_values, 
        X_display,
        plot_type="dot",
        color='k',
        cmap='Greys',
        show=False,
        max_display=10,
        alpha=0.5
    )
    
    # Set title
    plt.title('SHAP Summary Plot (Black & White)', fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'SHAP_beeswarm_BW.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Black and white beeswarm plot saved to: {output_path}")
    
    # Close figure
    plt.close()

def plot_shap_bar(shap_values, X, output_dir, config):
    """Plot SHAP bar chart (feature importance)"""
    # Rename features for display
    X_display = rename_features(X, config)
    
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Calculate mean absolute SHAP values for each feature
    feature_importance = np.abs(shap_values).mean(0)
    feature_names = X_display.columns
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    # Save feature importance data
    importance_path = os.path.join(output_dir, 'SHAP_feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importance data saved to: {importance_path}")
    
    # Plot bar chart
    shap.summary_plot(
        shap_values, 
        X_display,
        plot_type="bar",
        show=False,
        max_display=10
    )
    
    # Customize plot
    plt.title('SHAP Feature Importance', fontsize=16, fontweight='bold')
    plt.xlabel('mean(|SHAP value|)', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'SHAP_bar.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Bar chart saved to: {output_path}")
    
    # Close figure
    plt.close()

def plot_specific_dependence_plots(shap_values, X, output_dir, config):
    """Plot specific feature pair SHAP dependence plots"""
    # Get original exposure name and display name
    exposure = config['exposure']
    exposure_display = 'DII' if exposure == 'DII_food' else exposure
    
    # Rename features for display
    X_display = rename_features(X, config)
    
    # Plot DII and Age dependence
    plt.figure(figsize=(10, 7), dpi=300)
    
    # For dependence plots, we need to use original column names for indexing
    # but can customize the plot labels afterwards
    shap.dependence_plot(
        exposure, 
        shap_values, 
        X,
        interaction_index="Age",
        show=False
    )
    
    # Customize plot
    plt.title(f'SHAP Dependence Plot: {exposure_display} vs Age', fontsize=16, fontweight='bold')
    plt.xlabel(exposure_display, fontsize=12)
    plt.ylabel('SHAP value', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'SHAP_{exposure_display}_age.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"{exposure_display} vs Age dependence plot saved to: {output_path}")
    
    # Close figure
    plt.close()
    
    # Plot DII and BMI dependence
    plt.figure(figsize=(10, 7), dpi=300)
    shap.dependence_plot(
        exposure, 
        shap_values, 
        X,
        interaction_index="BMI",
        show=False
    )
    
    # Customize plot
    plt.title(f'SHAP Dependence Plot: {exposure_display} vs BMI', fontsize=16, fontweight='bold')
    plt.xlabel(exposure_display, fontsize=12)
    plt.ylabel('SHAP value', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'SHAP_{exposure_display}_BMI.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"{exposure_display} vs BMI dependence plot saved to: {output_path}")
    
    # Close figure
    plt.close()

def plot_top_dependence_plots(shap_values, X, output_dir, config):
    """Plot SHAP dependence plots for top features"""
    # Get original exposure name and display name
    exposure = config['exposure']
    exposure_display = 'DII' if exposure == 'DII_food' else exposure
    
    # Define feature name mapping for display
    feature_mapping = {
        'DII_food': 'DII',
        'ActivityLevel': 'PA'
    }
    
    # Calculate mean absolute SHAP values for each feature
    feature_importance = np.abs(shap_values).mean(0)
    feature_names = X.columns
    
    # Get top features by importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    # 保存特征重要性排名，便于解释选择标准
    importance_rank_path = os.path.join(output_dir, 'SHAP_feature_importance_rank.csv')
    importance_df.to_csv(importance_rank_path, index=False)
    print(f"Feature importance ranking saved to: {importance_rank_path}")
    
    # 获取前5个最重要的特征（或者所有特征，如果少于5个）
    top_n = min(5, len(feature_names))
    top_features = importance_df.head(top_n)['Feature'].values
    
    # Plot dependence plots for each important feature
    for feature in top_features:
        if feature == exposure:  # Already plotted specific dependence plots for exposure
            continue
            
        # Get display name for the feature
        feature_display = feature_mapping.get(feature, feature)
        
        plt.figure(figsize=(10, 6), dpi=300)
        
        # 寻找与当前特征交互最强的特征
        # 计算每个特征与当前特征的SHAP值相关性
        correlations = []
        for other_feature in feature_names:
            if other_feature != feature:
                # 获取特征索引
                idx_feature = np.where(X.columns == feature)[0][0]
                idx_other = np.where(X.columns == other_feature)[0][0]
                
                # 计算SHAP值的相关性
                corr = np.corrcoef(shap_values[:, idx_feature], shap_values[:, idx_other])[0, 1]
                correlations.append((other_feature, abs(corr)))
        
        # 选择相关性最强的特征作为交互特征
        if correlations:
            correlations.sort(key=lambda x: x[1], reverse=True)
            interaction_feature = correlations[0][0]
            interaction_strength = correlations[0][1]
            
            # 如果相关性太弱，则不使用交互特征
            if interaction_strength < 0.1:
                interaction_index = None
                interaction_title = ""
            else:
                interaction_index = interaction_feature
                interaction_display = feature_mapping.get(interaction_feature, interaction_feature)
                interaction_title = f" (with {interaction_display} interaction)"
        else:
            interaction_index = None
            interaction_title = ""
        
        # Plot dependence
        shap.dependence_plot(
            feature, 
            shap_values, 
            X,
            interaction_index=interaction_index,
            show=False,
            alpha=0.7,  # 降低点的透明度，使模式更明显
            dot_size=20  # 增大点的大小
        )
        
        # Customize plot
        plt.title(f'SHAP Dependence Plot: {feature_display}{interaction_title}', fontsize=16, fontweight='bold')
        plt.xlabel(feature_display, fontsize=12)
        plt.ylabel('SHAP value', fontsize=12)
        
        # 添加零线
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.6)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'SHAP_dependence_{feature_display}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Dependence plot for {feature_display} saved to: {output_path}")
        
        # Close figure
        plt.close()

def plot_waterfall_plots(explainer, X, output_dir, config):
    """Plot SHAP waterfall plots for selected examples"""
    # Rename features for display
    X_display = rename_features(X, config)
    
    # Define feature name mapping for display
    feature_mapping = {
        'DII_food': 'DII',
        'ActivityLevel': 'PA'
    }
    
    # Get a few interesting examples
    # 1. Select a high-risk case (high positive prediction)
    # 2. Select a borderline case (prediction close to 0.5)
    # 3. Select a low-risk case (low prediction)
    
    # Get model predictions
    try:
        # Try CatBoost prediction method
        if hasattr(explainer.model, 'predict_proba'):
            preds = explainer.model.predict_proba(X)[:, 1]
        elif hasattr(explainer.model, 'predict') and 'prediction_type' in explainer.model.predict.__code__.co_varnames:
            preds = explainer.model.predict(X, prediction_type='Probability')[:, 1]
        else:
            # Use SHAP values directly to estimate predictions
            # Base value + sum of SHAP values gives prediction in log-odds, convert to probability
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            
            # Get expected value (base value)
            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1]  # For binary classification
                
            # Calculate log odds and convert to probability
            log_odds = expected_value + shap_values.sum(axis=1)
            preds = 1 / (1 + np.exp(-log_odds))
    except:
        # Fallback: randomly select examples
        print("Warning: Could not get model predictions. Selecting random examples.")
        np.random.seed(42)  # For reproducibility
        preds = np.random.random(len(X))
    
    # Find indices of interesting examples
    high_risk_idx = np.argmax(preds)
    low_risk_idx = np.argmin(preds)
    
    # Find borderline case (closest to 0.5)
    borderline_idx = np.argmin(np.abs(preds - 0.5))
    
    # Create waterfall plots
    example_indices = {
        'high_risk': high_risk_idx,
        'borderline': borderline_idx,
        'low_risk': low_risk_idx
    }
    
    # Get expected value (base value)
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]  # For binary classification, get the positive class
    
    for name, idx in example_indices.items():
        plt.figure(figsize=(10, 8), dpi=300)
        
        # Get SHAP values for this instance
        shap_values_instance = explainer.shap_values(X.iloc[idx:idx+1])
        if isinstance(shap_values_instance, list):
            shap_values_instance = shap_values_instance[1][0]  # For binary classification
        else:
            shap_values_instance = shap_values_instance[0]
            
        # Create a mapping for feature display names
        feature_names = X.columns.tolist()
        display_names = [feature_mapping.get(name, name) for name in feature_names]
        
        # Plot waterfall
        shap.plots._waterfall.waterfall_legacy(
            expected_value, 
            shap_values_instance, 
            feature_names=display_names,
            max_display=10,
            show=False
        )
        
        # Add title
        pred_value = preds[idx]
        plt.title(f'SHAP Waterfall Plot - {name.replace("_", " ").title()} Example\nPrediction: {pred_value:.3f}', 
                  fontsize=16, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'SHAP_waterfall_{name}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Waterfall plot for {name} example saved to: {output_path}")
        
        # Close figure
        plt.close()

def plot_force_plots(explainer, X, output_dir, config):
    """Plot SHAP force plots for selected examples"""
    # Define feature name mapping for display
    feature_mapping = {
        'DII_food': 'DII',
        'ActivityLevel': 'PA'
    }
    
    # Get model predictions
    try:
        # Try CatBoost prediction method
        if hasattr(explainer.model, 'predict_proba'):
            preds = explainer.model.predict_proba(X)[:, 1]
        elif hasattr(explainer.model, 'predict') and 'prediction_type' in explainer.model.predict.__code__.co_varnames:
            preds = explainer.model.predict(X, prediction_type='Probability')[:, 1]
        else:
            # Use SHAP values directly to estimate predictions
            # Base value + sum of SHAP values gives prediction in log-odds, convert to probability
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            
            # Get expected value (base value)
            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1]  # For binary classification
                
            # Calculate log odds and convert to probability
            log_odds = expected_value + shap_values.sum(axis=1)
            preds = 1 / (1 + np.exp(-log_odds))
    except:
        # Fallback: randomly select examples
        print("Warning: Could not get model predictions. Selecting random examples.")
        np.random.seed(42)  # For reproducibility
        preds = np.random.random(len(X))
    
    # Find indices of interesting examples
    high_risk_idx = np.argmax(preds)
    low_risk_idx = np.argmin(preds)
    
    # Find borderline case (closest to 0.5)
    borderline_idx = np.argmin(np.abs(preds - 0.5))
    
    # Create force plots
    example_indices = {
        'high_risk': high_risk_idx,
        'borderline': borderline_idx,
        'low_risk': low_risk_idx
    }
    
    # Get expected value (base value)
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]  # For binary classification, get the positive class
    
    for name, idx in example_indices.items():
        # 使用更大的图形尺寸，特别是高度
        plt.figure(figsize=(14, 6), dpi=300)
        
        # Get SHAP values for this instance
        shap_values_instance = explainer.shap_values(X.iloc[idx:idx+1])
        if isinstance(shap_values_instance, list):
            shap_values_instance = shap_values_instance[1]  # For binary classification
        
        # Create a mapping for feature display names
        features = X.iloc[idx:idx+1].copy()
        for old_name, new_name in feature_mapping.items():
            if old_name in features.columns:
                features = features.rename(columns={old_name: new_name})
        
        # Plot force plot with improved layout
        shap.force_plot(
            expected_value, 
            shap_values_instance, 
            features,
            matplotlib=True,
            show=False,
            figsize=(14, 6),  # 增加高度
            text_rotation=45,  # 旋转文本以防重叠
            contribution_threshold=0.05  # 只显示重要的贡献
        )
        
        # Add title
        pred_value = preds[idx]
        plt.title(f'SHAP Force Plot - {name.replace("_", " ").title()} Example (Prediction: {pred_value:.3f})', 
                  fontsize=16, fontweight='bold')
        
        # Adjust layout with more space
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # 增加底部空间
        
        # Save figure
        output_path = os.path.join(output_dir, f'SHAP_force_{name}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Force plot for {name} example saved to: {output_path}")
        
        # Close figure
        plt.close()

def plot_interaction_heatmap(shap_values, X, output_dir, config):
    """Plot SHAP interaction heatmap"""
    # Get top features for interactions
    feature_importance = np.abs(shap_values).mean(0)
    feature_names = X.columns
    
    # Get top 6 most important features for interaction analysis
    top_features = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False).head(6)['Feature'].values
    
    # Define feature name mapping for display
    feature_mapping = {
        'DII_food': 'DII',
        'ActivityLevel': 'PA'
    }
    
    # Create a display name mapping for the top features
    display_names = [feature_mapping.get(f, f) for f in top_features]
    
    # Calculate pairwise interaction strengths using a more reliable method
    n_features = len(top_features)
    interaction_matrix = np.zeros((n_features, n_features))
    
    # Sample a subset of data for interaction calculation (for speed)
    sample_size = min(2000, len(X))
    sample_indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X.iloc[sample_indices]
    shap_values_sample = shap_values[sample_indices]
    
    # 使用更可靠的方法计算特征交互
    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                # 对角线设置为0
                interaction_matrix[i, j] = 0
            else:
                # 计算特征i和特征j的SHAP值乘积的平均值
                # 这是一种简单但有效的交互强度度量
                feature_i = top_features[i]
                feature_j = top_features[j]
                
                # 获取特征的索引
                idx_i = np.where(X.columns == feature_i)[0][0]
                idx_j = np.where(X.columns == feature_j)[0][0]
                
                # 计算交互强度
                interaction_strength = np.mean(shap_values_sample[:, idx_i] * shap_values_sample[:, idx_j])
                interaction_matrix[i, j] = interaction_strength
    
    # Plot heatmap with improved visualization
    plt.figure(figsize=(10, 8), dpi=300)
    
    # 创建自定义颜色映射，确保零值为白色
    cmap = plt.cm.RdBu_r
    
    # 找出绝对值最大的交互强度，以确保颜色映射对称
    max_abs_val = np.max(np.abs(interaction_matrix))
    
    # 创建热图
    im = plt.imshow(interaction_matrix, cmap=cmap, vmin=-max_abs_val, vmax=max_abs_val)
    
    # 添加颜色条
    cbar = plt.colorbar(im, label='Interaction Strength')
    
    # 设置刻度和标签
    plt.xticks(np.arange(n_features), display_names, rotation=45, ha='right')
    plt.yticks(np.arange(n_features), display_names)
    
    # 添加网格线，使单元格更清晰
    plt.grid(False)
    
    # 添加交互强度值标签
    for i in range(n_features):
        for j in range(n_features):
            if i != j:  # 跳过对角线
                plt.text(j, i, f'{interaction_matrix[i, j]:.3f}',
                        ha='center', va='center', 
                        color='black' if abs(interaction_matrix[i, j]) < max_abs_val/2 else 'white',
                        fontsize=9)
    
    # 修改标题为指定的标题
    plt.title('SHAP Feature Interaction Heatmap', 
              fontsize=14, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    output_path = os.path.join(output_dir, 'SHAP_interaction_heatmap.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Interaction heatmap saved to: {output_path}")
    
    # 保存交互数据为CSV，以便进一步分析
    interaction_df = pd.DataFrame(
        interaction_matrix,
        index=display_names,
        columns=display_names
    )
    interaction_csv_path = os.path.join(output_dir, 'SHAP_interaction_matrix.csv')
    interaction_df.to_csv(interaction_csv_path)
    print(f"Interaction matrix saved to: {interaction_csv_path}")
    
    # 关闭图形
    plt.close()

def plot_quartile_analysis(shap_values, X, output_dir, config):
    """Plot SHAP values by quartiles for the exposure variable"""
    # Get exposure variable
    exposure = config['exposure']
    exposure_display = 'DII' if exposure == 'DII_food' else exposure
    
    # Calculate quartiles for the exposure
    X_quartiles = pd.qcut(X[exposure], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    # Get mean SHAP values for each feature by quartile
    feature_names = X.columns
    n_features = len(feature_names)
    quartile_shap = {}
    
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        # Get indices for this quartile
        q_indices = np.where(X_quartiles == q)[0]
        
        # Calculate mean SHAP values for each feature in this quartile
        quartile_shap[q] = np.mean(shap_values[q_indices], axis=0)
    
    # Define feature name mapping for display
    feature_mapping = {
        'DII_food': 'DII',
        'ActivityLevel': 'PA'
    }
    
    # Get top features
    mean_importance = np.abs(shap_values).mean(0)
    top_indices = np.argsort(-mean_importance)[:8]  # Top 8 features
    
    # Create a plot
    plt.figure(figsize=(12, 8), dpi=300)
    
    # Set up colors
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    
    # Set width of bars
    bar_width = 0.2
    
    # Set positions for bars
    r = np.arange(len(top_indices))
    
    # Plot bars for each quartile
    for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        plt.bar(r + i*bar_width, 
                quartile_shap[q][top_indices], 
                width=bar_width, 
                color=colors[i], 
                label=f'{exposure_display} {q}')
    
    # Add feature names
    display_names = [feature_mapping.get(feature_names[i], feature_names[i]) for i in top_indices]
    plt.xticks(r + bar_width*1.5, display_names, rotation=45, ha='right')
    
    # Add labels and legend
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Mean SHAP Value', fontsize=12)
    plt.title(f'Mean SHAP Values by {exposure_display} Quartile', fontsize=16, fontweight='bold')
    plt.legend()
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'SHAP_{exposure_display}_quartile_analysis.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Quartile analysis plot saved to: {output_path}")
    
    # Close figure
    plt.close()

def main():
    """Main function"""
    # Load configuration
    config = load_config()
    
    # Create output directory
    output_dir = 'plot_general'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    X, y, weights, dii_q = load_data(config)
    
    # Load model
    model = load_model()
    
    # Create SHAP explainer
    explainer = create_shap_explainer(model, X)
    
    # Get stratified sample indices
    summary_indices = get_sample_indices(y, dii_q)
    X_summary = X.iloc[summary_indices]
    
    # Calculate SHAP values for sampled data
    print("Calculating SHAP values for sampled data...")
    shap_values_summary = calculate_shap_values(explainer, X_summary)
    
    # Plot color SHAP beeswarm
    print("Plotting color SHAP beeswarm...")
    plot_shap_beeswarm(shap_values_summary, X_summary, output_dir, config)
    
    # Plot black and white SHAP beeswarm
    print("Plotting black and white SHAP beeswarm...")
    plot_shap_beeswarm_bw(shap_values_summary, X_summary, output_dir, config)
    
    # Plot SHAP bar chart (feature importance)
    print("Plotting SHAP bar chart...")
    plot_shap_bar(shap_values_summary, X_summary, output_dir, config)
    
    # Calculate SHAP values for all data (for dependence plots)
    print("Calculating SHAP values for all data (for dependence plots)...")
    shap_values = calculate_shap_values(explainer, X)
    
    # Plot specific feature pair SHAP dependence plots
    print("Plotting specific feature pair SHAP dependence plots...")
    plot_specific_dependence_plots(shap_values, X, output_dir, config)
    
    # Plot top feature SHAP dependence plots
    print("Plotting top feature SHAP dependence plots...")
    plot_top_dependence_plots(shap_values, X, output_dir, config)
    
    # Plot waterfall plots for selected examples
    print("Plotting waterfall plots...")
    plot_waterfall_plots(explainer, X, output_dir, config)
    
    # Plot force plots for selected examples
    print("Plotting force plots...")
    plot_force_plots(explainer, X, output_dir, config)
    
    # Plot interaction heatmap
    print("Plotting interaction heatmap...")
    plot_interaction_heatmap(shap_values, X, output_dir, config)
    
    # Plot quartile analysis
    print("Plotting quartile analysis...")
    plot_quartile_analysis(shap_values, X, output_dir, config)
    
    print("SHAP analysis complete! All plots saved to the plot directory.")

if __name__ == "__main__":
    main()
