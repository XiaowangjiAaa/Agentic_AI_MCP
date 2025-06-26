import os
import pandas as pd
import numpy as np
#from mcp.server.fastmcp import tool
from MCP.tool import tool

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 可替换为真实装饰器
def tool(name=None):
    def decorator(fn):
        return fn
    return decorator


@tool(name="compare_results_csv")
def compare_results_csv(gt_csv_path: str, pred_csv_path: str) -> dict:
    """
    多图像指标对比：从 GT 和预测 CSV 中读取指标，计算统计误差。
    包括 MAE, MSE, R², MAPE, 平均相对误差（百分比）。
    """
    try:
        if not os.path.exists(gt_csv_path) or not os.path.exists(pred_csv_path):
            raise FileNotFoundError("GT 或预测 CSV 文件不存在")

        df_gt = pd.read_csv(gt_csv_path).set_index("Image")
        df_pred = pd.read_csv(pred_csv_path).set_index("Image")

        # 对齐图像索引
        df = df_gt.join(df_pred, lsuffix="_gt", rsuffix="_pred").dropna()

        if df.empty:
            raise ValueError("无可比较图像，GT 与预测不匹配")

        metrics = {}
        for col in df_gt.columns:
            gt_vals = df[f"{col}_gt"]
            pred_vals = df[f"{col}_pred"]

            mae = mean_absolute_error(gt_vals, pred_vals)
            mse = mean_squared_error(gt_vals, pred_vals)
            r2 = r2_score(gt_vals, pred_vals)
            mape = np.mean(np.abs((gt_vals - pred_vals) / gt_vals)) * 100
            avg_rel_err = np.mean(np.abs(gt_vals - pred_vals) / gt_vals) * 100

            metrics[col] = {
                "MAE": round(mae, 4),
                "MSE": round(mse, 4),
                "R2": round(r2, 4),
                "MAPE (%)": round(mape, 2),
                "AvgRelError (%)": round(avg_rel_err, 2)
            }

        return {
            "status": "success",
            "summary": f"对比完成，处理图像数: {len(df)}",
            "outputs": metrics,
            "error": None
        }

    except Exception as e:
        return {
            "status": "error",
            "summary": "对比失败",
            "outputs": None,
            "error": str(e)
        }
