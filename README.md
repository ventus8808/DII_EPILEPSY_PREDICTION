# DII_EPILEPSY_PREDICTION

一个基于饮食炎症指数(DII)预测癫痫风险的机器学习项目。本项目利用NHANES数据集，通过多种机器学习模型探索DII与癫痫之间的关联，并评估模型性能。

## 项目概述

本项目旨在研究饮食炎症指数(DII)与癫痫风险之间的关系，通过多种机器学习算法构建预测模型，并进行全面的模型评估和比较。项目特点包括：

- 多种机器学习模型的训练与评估
- 全面的模型性能评估指标
- 详细的可视化分析
- 特征贡献度分析（尤其是DII的贡献）
- 模型校准与决策曲线分析
- 集成学习方法的应用

## 数据集

项目使用NHANES(美国国家健康与营养调查)数据集，包含以下关键变量：

- **结局变量**：癫痫状态(Epilepsy)
- **主要暴露因素**：饮食炎症指数(DII_food)
- **协变量**：性别(Gender)、年龄(Age)、体重指数(BMI)、教育水平(Education)、婚姻状况(Marriage)、吸烟状况(Smoke)、饮酒状况(Alcohol)、就业状况(Employment)、身体活动水平(ActivityLevel)

数据文件位于`data`目录下，主要使用`16_ML.csv`进行模型训练和评估。

## 模型

项目实现了以下机器学习模型：

1. **逻辑回归(Logistic Regression)**：基础线性模型，提供可解释的特征权重
2. **随机森林(Random Forest, RF)**：基于决策树的集成学习方法
3. **XGBoost**：梯度提升树算法，高效处理非线性关系
4. **LightGBM**：微软开发的梯度提升框架，速度更快
5. **CatBoost**：处理类别特征的梯度提升算法
6. **高斯朴素贝叶斯(Gaussian Naive Bayes, GNB)**：基于贝叶斯定理的概率模型
7. **前馈神经网络(Feed-forward Neural Network, FNN)**：使用MLX框架实现的神经网络模型
8. **集成投票分类器(Ensemble Voting)**：结合多个基模型的投票集成方法

每个模型都经过了超参数优化，使用Optuna框架进行参数搜索。

## 项目结构

```
DII_EPILEPSY_PREDICTION/
├── config.yaml              # 项目配置文件
├── data/                    # 数据目录
│   └── 16_ML.csv            # 主要数据文件
├── model/                   # 训练好的模型和参数
├── py_model/                # 模型训练和评估代码
│   ├── *_Train.py           # 各模型训练脚本
│   ├── *_Evaluation.py      # 各模型评估脚本
│   └── model_*.py           # 模型工具函数
├── py_general/              # 通用功能代码
│   ├── distribution_DII.py  # DII分布分析
│   ├── combine_*.py         # 结果合并脚本
│   └── plot_*.py            # 绘图脚本
├── plot/                    # 单模型图表
├── plot_combined/           # 组合对比图表
├── plot_distribution/       # 分布图表
├── plot_original_data/      # 图表原始数据
└── result/                  # 结果输出目录
```

## 功能特点

### 1. 模型训练

- **超参数优化**：使用Optuna框架进行贝叶斯优化
- **交叉验证**：使用分层k折交叉验证评估模型性能
- **SMOTE过采样**：处理类别不平衡问题
- **多目标优化**：同时优化多个性能指标(AUC、精确度、敏感度等)
- **硬约束条件**：设置模型性能的最低要求

### 2. 模型评估

- **分类性能指标**：准确率、精确度、敏感度、特异度、F1分数、AUC-ROC、AUC-PR等
- **校准评估**：校准曲线、期望校准误差(ECE)
- **决策曲线分析(DCA)**：评估模型在不同决策阈值下的临床净收益
- **混淆矩阵**：评估分类性能
- **学习曲线**：评估模型随训练样本量增加的性能变化
- **阈值曲线**：评估不同决策阈值对模型性能的影响

### 3. 特征分析

- **特征重要性**：评估各特征对预测的贡献
- **DII贡献分析**：通过比较含DII和不含DII的模型性能，评估DII的预测贡献
- **特征交互作用**：分析特征间的交互效应

### 4. 可视化

- **ROC曲线**：评估模型的区分能力
- **PR曲线**：评估模型在类别不平衡情况下的性能
- **校准曲线**：评估预测概率的准确性
- **DCA曲线**：评估临床决策价值
- **学习曲线**：评估样本量对性能的影响
- **阈值曲线**：评估决策阈值的影响
- **DII分布分析**：评估DII的分布特性和正态性

## 使用方法

### 环境配置

项目依赖以下主要库：
- Python 3.8+
- pandas, numpy
- scikit-learn
- xgboost, lightgbm, catboost
- mlx (用于神经网络)
- optuna (用于超参数优化)
- matplotlib, seaborn (用于可视化)

### 模型训练

1. 修改`config.yaml`配置文件，设置数据路径、特征、目标变量等
2. 运行模型训练脚本：
   ```
   python py_model/XGBoost_Train.py
   ```

### 模型评估

1. 运行模型评估脚本：
   ```
   python py_model/XGBoost_Evaluation.py
   ```
2. 可以通过命令行参数控制评估内容，例如：
   ```
   python py_model/XGBoost_Evaluation.py --roc 1 --pr 1 --calibration 1 --dca 1
   ```

### 结果可视化

1. 生成组合图表：
   ```
   python py_general/combine_plots_all.py
   ```
2. 分析DII分布：
   ```
   python py_general/distribution_DII.py
   ```

## 主要结果

1. 所有模型均显示DII与癫痫风险存在关联
2. 集成模型(Voting)在大多数评估指标上表现最佳
3. 在单模型中，XGBoost和CatBoost表现较好
4. DII对预测性能有显著贡献，移除DII特征会导致模型性能下降
5. 模型校准良好，预测概率与实际风险相符

## 未来工作

1. 纳入更多潜在的混杂因素
2. 探索更复杂的特征工程方法
3. 实现深度学习模型
4. 开发交互式预测工具
5. 进行外部验证研究

## 联系方式

如有任何问题或建议，请联系项目维护者。 