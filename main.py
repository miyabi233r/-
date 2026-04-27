# main.py
import os
import time
from src.data_processing import load_data, clean_data, feature_engineering, save_cleaned_data, \
    generate_data_quality_report
from src.analysis import load_cleaned_data as load_for_analysis
from src.analysis import plot_demand_by_time, plot_region_heatmap, plot_fare_factors, plot_custom_insight
from src.model import load_cleaned_data as load_for_model, prepare_prediction_data, build_and_train_models
from src.qa import simple_qa_system


def main():
    print("=" * 80)
    print("   《人工智能编程语言》期末大作业 - 纽约出租车智能问答系统")
    print("   作者：任雅文  学号：23354131")
    print("=" * 80)
    print("开始执行完整流程...\n")

    start_time = time.time()

    # M1: 数据处理
    print("【M1】数据加载与处理...")
    df, zones = load_data()
    generate_data_quality_report(df)
    df_clean = clean_data(df)
    df_clean = feature_engineering(df_clean, zones)
    save_cleaned_data(df_clean)
    print("M1 完成！\n")

    # M2: 分析可视化
    print("【M2】分析与可视化（生成4张图表）...")
    df_vis = load_for_analysis()
    plot_demand_by_time(df_vis)
    plot_region_heatmap(df_vis)
    plot_fare_factors(df_vis)
    plot_custom_insight(df_vis)
    print("M2 完成！所有图表已保存至 outputs/ 目录\n")

    # M3: 预测模型
    print("【M3】训练预测模型（神经网络 + 随机森林）...")
    df_model = load_for_model()
    demand_df = prepare_prediction_data(df_model)
    build_and_train_models(demand_df)
    print("M3 完成！模型对比与 Loss 曲线已生成\n")

    # M4: 问答系统
    print("【M4】启动智能问答接口...")
    print("系统已就绪！\n")

    total_time = time.time() - start_time
    print(f"全流程执行完成！总耗时: {total_time / 60:.1f} 分钟")
    print("现在进入问答模式...\n")

    # 启动问答循环
    simple_qa_system()


if __name__ == "__main__":
    # 创建必要目录
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    main()