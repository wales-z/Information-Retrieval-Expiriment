# Information-Retrieval-Expiriment
UESTC，信息检索课程实验

## 目录说明：

- main.py，我自己实现梯度下降的代码（包含实验主流程代码）
- main_mf.py，调用深度学习库实现梯度下降的对比实验的代码

- model.py，包含：

  - mfDataset类，用于将预处理好的数据集变量转变成 pytorch 训练用的数据集对象，常与Dataloader配合使用
  - mf类，是用pytorch实现的矩阵分解模型类
  - compute_rating函数，是我自己实现梯度下降的代码中计算预测评分要用的函数。

- 其他 Jupyter notebook文件是写代码时为了方便调试而写的，可以不用管。

- IR_expiriment_0.log，未添加l2正则化时的实验结果

- IR_expiriment_1.log，添加l2正则化后，10个epoch的结果

- IR_expiriment_1.log，添加l2正则化后，20个epoch的结果（实验还没跑完，所以是空的）

- ml-1m：MovieLens-1M数据集。

  

