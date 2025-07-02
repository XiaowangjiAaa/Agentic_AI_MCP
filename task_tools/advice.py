from task_tools.registry import tool
import pandas as pd
import os

@tool(name="summarize_and_advice")
def summarize_and_advice(results_csv: str = "outputs/csv/predicted_metrics.csv") -> dict:
    """Summarize prediction metrics and give simple advice."""
    try:
        if not os.path.exists(results_csv):
            raise FileNotFoundError(f"Results CSV not found: {results_csv}")
        df = pd.read_csv(results_csv)
        if df.empty:
            raise ValueError("Results CSV is empty")
        averages = df.mean(numeric_only=True).to_dict()
        summary = (
            f"平均长度 {averages.get('Length (mm)', 0):.2f}mm, "
            f"平均面积 {averages.get('Area (mm^2)', 0):.2f}mm^2, "
            f"平均最大宽度 {averages.get('Max Width (mm)', 0):.2f}mm"
        )
        return {
            "status": "success",
            "summary": summary,
            "outputs": averages,
            "error": None,
        }
    except Exception as e:
        return {
            "status": "error",
            "summary": "汇总失败",
            "outputs": None,
            "error": str(e),
        }
