"""
This script is used for computing fairness metrics from the given input csv.
"""

import argparse

import numpy as np
import pandas as pd
from icecream import ic
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm
import time


def load_data(dataset_name, llm_name):
    # Placeholder function to load data based on dataset and LLM names
    # In a real implementation, this would load the appropriate data

    if dataset_name == "mimic":
        path = f"/data/zikangxu/Documents/medrax/fairness_eval/logs/{dataset_name}/analysis/fairness/{llm_name}/fairness_posthoc/per_question_features.csv"

    elif dataset_name == "chexagentbench":
        path = f"/data/zikangxu/Documents/medrax/fairness_eval/logs/{dataset_name}/analysis/{llm_name}/fairness_posthoc/per_question_features.csv"

    dataframe = pd.read_csv(path)
    # ic(dataframe.columns)

    # only preserve the columns that are needed for computing fairness metrics
    columns_to_keep = ["predicted_answer",
                       "correct_answer", "age_raw", "gender"]
    dataframe = dataframe[columns_to_keep]

    # turn attr to binary values

    dataframe['age_raw'] = (dataframe['age_raw'] >= 60).astype(int)

    dataframe['gender'] = (
        dataframe['gender']
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(['m', 'male'])
        .astype(int)
    )
    # ic(dataframe.head())
    return dataframe


def compute_fairness_metrics(dataframe):
    """
    Dataframe:
    predicted_answer: the answer predicted by the LLM
    correct_answer: the ground truth answer
    age_raw: the age of the user (binary: 1 for 60 and above, 0 for below 60)
    gender: the gender of the user (binary: 1 for male, 0 for female)

    Fairness Metrics to compute:
    1. ACC
    2. Delta-ACC
    3. DP
    4. EoD
    5. FUT
    """
    metrics = {}

    # 1. Overall Accuracy (ACC)
    correct = (dataframe["predicted_answer"] ==
               dataframe["correct_answer"]).astype(int)
    overall_acc = correct.sum() / len(dataframe)
    metrics["ACC"] = overall_acc

    # 2. Delta-ACC (Accuracy difference across demographic groups)

    delta_acc = {}
    for attr in ["age_raw", "gender"]:
        acc_0 = correct[dataframe[attr] == 0].mean()
        acc_1 = correct[dataframe[attr] == 1].mean()
        delta_acc[attr] = abs(acc_0 - acc_1)
    metrics["Delta-ACC"] = delta_acc

    # 3. Demographic Parity (DP)
    # For multiclass: treat each class as positive, others as negative, then average
    dp = {}
    unique_classes = pd.concat(
        [dataframe["predicted_answer"], dataframe["correct_answer"]]).unique()

    for attr in ["age_raw", "gender"]:
        dp_scores = []
        for class_label in unique_classes:
            # Treat current class as positive, others as negative
            is_correct = (dataframe["predicted_answer"] == class_label) & (
                dataframe["correct_answer"] == class_label)
            is_positive_pred = dataframe["predicted_answer"] == class_label

            # Positive prediction rate for each group
            group_0_mask = dataframe[attr] == 0
            group_1_mask = dataframe[attr] == 1

            pred_pos_rate_0 = is_positive_pred[group_0_mask].mean(
            ) if group_0_mask.sum() > 0 else 0
            pred_pos_rate_1 = is_positive_pred[group_1_mask].mean(
            ) if group_1_mask.sum() > 0 else 0

            dp_scores.append(abs(pred_pos_rate_0 - pred_pos_rate_1))

        dp[attr] = max(dp_scores)
    metrics["DP"] = dp

    # 4. Equalized Odds Difference (EoD)
    # For multiclass: treat each class as positive, others as negative, then average
    eod = {}
    unique_classes = pd.concat(
        [dataframe["predicted_answer"], dataframe["correct_answer"]]).unique()

    for attr in ["age_raw", "gender"]:
        tpr_diffs = []
        fpr_diffs = []

        for class_label in unique_classes:
            # Binary classification: current class as positive, others as negative
            is_correct_class = dataframe["correct_answer"] == class_label
            is_pred_class = dataframe["predicted_answer"] == class_label

            group_0_mask = dataframe[attr] == 0
            group_1_mask = dataframe[attr] == 1

            # True Positive Rate (TPR): correctly predicted positive among actual positives
            tp_0 = ((is_pred_class) & (is_correct_class) & group_0_mask).sum()
            actual_pos_0 = is_correct_class[group_0_mask].sum()
            tpr_0 = tp_0 / actual_pos_0 if actual_pos_0 > 0 else 0

            tp_1 = ((is_pred_class) & (is_correct_class) & group_1_mask).sum()
            actual_pos_1 = is_correct_class[group_1_mask].sum()
            tpr_1 = tp_1 / actual_pos_1 if actual_pos_1 > 0 else 0

            tpr_diffs.append(abs(tpr_0 - tpr_1))

            # False Positive Rate (FPR): incorrectly predicted positive among actual negatives
            is_not_correct_class = dataframe["correct_answer"] != class_label
            fp_0 = ((is_pred_class) & (is_not_correct_class)
                    & group_0_mask).sum()
            actual_neg_0 = is_not_correct_class[group_0_mask].sum()
            fpr_0 = fp_0 / actual_neg_0 if actual_neg_0 > 0 else 0

            fp_1 = ((is_pred_class) & (is_not_correct_class)
                    & group_1_mask).sum()
            actual_neg_1 = is_not_correct_class[group_1_mask].sum()
            fpr_1 = fp_1 / actual_neg_1 if actual_neg_1 > 0 else 0

            fpr_diffs.append(abs(fpr_0 - fpr_1))

        avg_tpr_diff = max(tpr_diffs)if len(tpr_diffs) > 0 else 0
        avg_fpr_diff = max(fpr_diffs) if len(fpr_diffs) > 0 else 0

        eod[attr] = {
            "TPR_diff": avg_tpr_diff,
            "FPR_diff": avg_fpr_diff,
            "EoD": (avg_tpr_diff + avg_fpr_diff) / 2
        }
    metrics["EoD"] = eod

    # 5. Fairness-utility Tradeoff(FUT)
    '''
    FUT = meanACC / (1+std(ACC) for each group
    '''
    fut = {}
    for attr in ["age_raw", "gender"]:
        acc_0 = correct[dataframe[attr] == 0].mean()
        acc_1 = correct[dataframe[attr] == 1].mean()
        mean_acc = (acc_0 + acc_1) / 2
        std_acc = ((acc_0 - mean_acc) ** 2 + (acc_1 - mean_acc) ** 2) ** 0.5
        fut[attr] = mean_acc / (1 + std_acc)
    metrics["FUT"] = fut

    return metrics


def format_result(llm, metrics_dict):

    latex_output = f"\\textbf{{{llm}}} & ${metrics_dict['ACC']*100:.2f}$ & "
    latex_output += f"${metrics_dict['Delta-ACC']['gender']*100:.2f}$ & "
    latex_output += f"${metrics_dict['DP']['gender']*100:.2f}$ & "
    latex_output += f"${metrics_dict['EoD']['gender']['EoD']*100:.2f}$ & "
    latex_output += f"${metrics_dict['FUT']['gender']*100:.2f}$"
    # ic(latex_output)

    latex_output = f"\\textbf{{{llm}}} & ${metrics_dict['ACC']*100:.2f}$ & "
    latex_output += f"${metrics_dict['Delta-ACC']['age_raw']*100:.2f}$ & "
    latex_output += f"${metrics_dict['DP']['age_raw']*100:.2f}$ & "
    latex_output += f"${metrics_dict['EoD']['age_raw']['EoD']*100:.2f}$ & "
    latex_output += f"${metrics_dict['FUT']['age_raw']*100:.2f}$"
    # ic(latex_output)
    # Placeholder function to format the metrics dictionary into a readable format
    # In a real implementation, this would format the results as needed
    return latex_output


def bootstrap_metrics(dataframe, n_iterations=1000, confidence_level=0.95):
    """
    对 dataframe 进行 bootstrap 重采样，计算指标的 mean 和 std
    """
    from tqdm import tqdm

    bootstrap_results = []

    for i in tqdm(range(n_iterations), desc="Bootstrap sampling"):
        # 有放回抽样
        sample = dataframe.sample(n=len(dataframe), replace=True)
        metrics = compute_fairness_metrics(sample)
        bootstrap_results.append(metrics)

    # 计算 mean 和 std
    '''
    1. ACC
    2. Delta-ACC (gender, age_raw)
    3. DP (gender, age_raw)
    4. EoD (gender, age_raw)
    5. FUT (gender, age_raw)
    '''
    summary = {}

    # 计算置信区间的百分位数
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    # ACC
    acc_values = [r['ACC'] for r in bootstrap_results]
    summary['ACC'] = {
        'mean': np.mean(acc_values),
        'std': np.std(acc_values),
        'c_low': np.percentile(acc_values, lower_percentile),
        'c_high': np.percentile(acc_values, upper_percentile)
    }

    # Delta-ACC
    delta_acc_gender_values = [r['Delta-ACC']['gender']
                               for r in bootstrap_results]
    delta_acc_age_values = [r['Delta-ACC']['age_raw']
                            for r in bootstrap_results]
    summary['Delta-ACC'] = {
        'gender': {
            'mean': np.mean(delta_acc_gender_values),
            'std': np.std(delta_acc_gender_values),
            'c_low': np.percentile(delta_acc_gender_values, lower_percentile),
            'c_high': np.percentile(delta_acc_gender_values, upper_percentile)
        },
        'age_raw': {
            'mean': np.mean(delta_acc_age_values),
            'std': np.std(delta_acc_age_values),
            'c_low': np.percentile(delta_acc_age_values, lower_percentile),
            'c_high': np.percentile(delta_acc_age_values, upper_percentile)
        }
    }

    # DP
    dp_gender_values = [r['DP']['gender'] for r in bootstrap_results]
    dp_age_values = [r['DP']['age_raw'] for r in bootstrap_results]
    summary['DP'] = {
        'gender': {
            'mean': np.mean(dp_gender_values),
            'std': np.std(dp_gender_values),
            'c_low': np.percentile(dp_gender_values, lower_percentile),
            'c_high': np.percentile(dp_gender_values, upper_percentile)
        },
        'age_raw': {
            'mean': np.mean(dp_age_values),
            'std': np.std(dp_age_values),
            'c_low': np.percentile(dp_age_values, lower_percentile),
            'c_high': np.percentile(dp_age_values, upper_percentile)
        }
    }

    # EoD
    eod_gender_values = [r['EoD']['gender']['EoD'] for r in bootstrap_results]
    eod_age_values = [r['EoD']['age_raw']['EoD'] for r in bootstrap_results]
    summary['EoD'] = {
        'gender': {
            'mean': np.mean(eod_gender_values),
            'std': np.std(eod_gender_values),
            'c_low': np.percentile(eod_gender_values, lower_percentile),
            'c_high': np.percentile(eod_gender_values, upper_percentile)
        },
        'age_raw': {
            'mean': np.mean(eod_age_values),
            'std': np.std(eod_age_values),
            'c_low': np.percentile(eod_age_values, lower_percentile),
            'c_high': np.percentile(eod_age_values, upper_percentile)
        }
    }

    # FUT
    fut_gender_values = [r['FUT']['gender'] for r in bootstrap_results]
    fut_age_values = [r['FUT']['age_raw'] for r in bootstrap_results]
    summary['FUT'] = {
        'gender': {
            'mean': np.mean(fut_gender_values),
            'std': np.std(fut_gender_values),
            'c_low': np.percentile(fut_gender_values, lower_percentile),
            'c_high': np.percentile(fut_gender_values, upper_percentile)
        },
        'age_raw': {
            'mean': np.mean(fut_age_values),
            'std': np.std(fut_age_values),
            'c_low': np.percentile(fut_age_values, lower_percentile),
            'c_high': np.percentile(fut_age_values, upper_percentile)
        }
    }

    return summary, bootstrap_results


def format_bootstrap_result(llm, bootstrap_summary):
    """
    Format bootstrap results as mean [c_low, c_high]
    """
    # print(f"\n{'='*70}")
    # print(f"Bootstrap Results for {llm} (mean [c_low, c_high])")
    # print(f"{'='*70}\n")

    # Gender-based metrics
    # print("Gender-based metrics:")
    # print("-" * 70)
    acc_mean = bootstrap_summary['ACC']['mean'] * 100
    acc_c_low = bootstrap_summary['ACC']['c_low'] * 100
    acc_c_high = bootstrap_summary['ACC']['c_high'] * 100
    delta_acc_mean = bootstrap_summary['Delta-ACC']['gender']['mean'] * 100
    delta_acc_c_low = bootstrap_summary['Delta-ACC']['gender']['c_low'] * 100
    delta_acc_c_high = bootstrap_summary['Delta-ACC']['gender']['c_high'] * 100
    dp_mean = bootstrap_summary['DP']['gender']['mean'] * 100
    dp_c_low = bootstrap_summary['DP']['gender']['c_low'] * 100
    dp_c_high = bootstrap_summary['DP']['gender']['c_high'] * 100
    eod_mean = bootstrap_summary['EoD']['gender']['mean'] * 100
    eod_c_low = bootstrap_summary['EoD']['gender']['c_low'] * 100
    eod_c_high = bootstrap_summary['EoD']['gender']['c_high'] * 100
    fut_mean = bootstrap_summary['FUT']['gender']['mean'] * 100
    fut_c_low = bootstrap_summary['FUT']['gender']['c_low'] * 100
    fut_c_high = bootstrap_summary['FUT']['gender']['c_high'] * 100

    # print(f"ACC:       {acc_mean:.2f} [{acc_c_low:.2f}, {acc_c_high:.2f}]")
    # print(f"Delta-ACC: {delta_acc_mean:.2f} [{delta_acc_c_low:.2f}, {delta_acc_c_high:.2f}]")
    # print(f"DP:        {dp_mean:.2f} [{dp_c_low:.2f}, {dp_c_high:.2f}]")
    # print(f"EoD:       {eod_mean:.2f} [{eod_c_low:.2f}, {eod_c_high:.2f}]")
    # print(f"FUT:       {fut_mean:.2f} [{fut_c_low:.2f}, {fut_c_high:.2f}]")

    latex_gender = f"\\textbf{{{llm}}} & "
    latex_gender += f"${acc_mean:.2f}_{{[{acc_c_low:.2f}, {acc_c_high:.2f}]}}$ & "
    latex_gender += f"${delta_acc_mean:.2f}_{{[{delta_acc_c_low:.2f}, {delta_acc_c_high:.2f}]}}$ & "
    latex_gender += f"${dp_mean:.2f}_{{[{dp_c_low:.2f}, {dp_c_high:.2f}]}}$ & "
    latex_gender += f"${eod_mean:.2f}_{{[{eod_c_low:.2f}, {eod_c_high:.2f}]}}$ & "
    latex_gender += f"${fut_mean:.2f}_{{[{fut_c_low:.2f}, {fut_c_high:.2f}]}}$ \\"
    # print(f"\nLaTeX (Gender):\n{latex_gender}\n")

    # Age-based metrics
    # print("Age-based metrics:")
    # print("-" * 70)
    delta_acc_mean = bootstrap_summary['Delta-ACC']['age_raw']['mean'] * 100
    delta_acc_c_low = bootstrap_summary['Delta-ACC']['age_raw']['c_low'] * 100
    delta_acc_c_high = bootstrap_summary['Delta-ACC']['age_raw']['c_high'] * 100
    dp_mean = bootstrap_summary['DP']['age_raw']['mean'] * 100
    dp_c_low = bootstrap_summary['DP']['age_raw']['c_low'] * 100
    dp_c_high = bootstrap_summary['DP']['age_raw']['c_high'] * 100
    eod_mean = bootstrap_summary['EoD']['age_raw']['mean'] * 100
    eod_c_low = bootstrap_summary['EoD']['age_raw']['c_low'] * 100
    eod_c_high = bootstrap_summary['EoD']['age_raw']['c_high'] * 100
    fut_mean = bootstrap_summary['FUT']['age_raw']['mean'] * 100
    fut_c_low = bootstrap_summary['FUT']['age_raw']['c_low'] * 100
    fut_c_high = bootstrap_summary['FUT']['age_raw']['c_high'] * 100

    # print(f"ACC:       {acc_mean:.2f} [{acc_c_low:.2f}, {acc_c_high:.2f}]")
    # print(f"Delta-ACC: {delta_acc_mean:.2f} [{delta_acc_c_low:.2f}, {delta_acc_c_high:.2f}]")
    # print(f"DP:        {dp_mean:.2f} [{dp_c_low:.2f}, {dp_c_high:.2f}]")
    # print(f"EoD:       {eod_mean:.2f} [{eod_c_low:.2f}, {eod_c_high:.2f}]")
    # print(f"FUT:       {fut_mean:.2f} [{fut_c_low:.2f}, {fut_c_high:.2f}]")

    latex_age = f"& "
    latex_age += f"${acc_mean:.2f}_{{[{acc_c_low:.2f}, {acc_c_high:.2f}]}}$ & "
    latex_age += f"${delta_acc_mean:.2f}_{{[{delta_acc_c_low:.2f}, {delta_acc_c_high:.2f}]}}$ & "
    latex_age += f"${dp_mean:.2f}_{{[{dp_c_low:.2f}, {dp_c_high:.2f}]}}$ & "
    latex_age += f"${eod_mean:.2f}_{{[{eod_c_low:.2f}, {eod_c_high:.2f}]}}$ & "
    latex_age += f"${fut_mean:.2f}_{{[{fut_c_low:.2f}, {fut_c_high:.2f}]}}$ \\\\"
    # print(f"\nLaTeX (Age):\n{latex_age}\n")
    # print(f"{'='*70}\n")

    return {'gender': latex_gender, 'age': latex_age}


def format_multiple_bootstrap_results(llm_results_dict, attribute='gender'):
    """
    Format multiple LLM results and mark best/second-best with \\xbest{} and \\xsecond{}
    
    Args:
        llm_results_dict: Dict mapping LLM names to their bootstrap_summary
        attribute: 'gender' or 'age_raw'
    
    Returns:
        Dict mapping LLM names to formatted LaTeX strings
    """
    # 对于每个指标，确定哪个是最好/次好的
    # ACC: 越高越好
    # Delta-ACC, DP, EoD: 越低越好
    # FUT: 越高越好
    
    llm_names = list(llm_results_dict.keys())
    
    # 收集所有LLM的指标值（使用mean）
    acc_values = {llm: llm_results_dict[llm]['ACC']['mean'] * 100 for llm in llm_names}
    delta_acc_values = {llm: llm_results_dict[llm]['Delta-ACC'][attribute]['mean'] * 100 for llm in llm_names}
    dp_values = {llm: llm_results_dict[llm]['DP'][attribute]['mean'] * 100 for llm in llm_names}
    eod_values = {llm: llm_results_dict[llm]['EoD'][attribute]['mean'] * 100 for llm in llm_names}
    fut_values = {llm: llm_results_dict[llm]['FUT'][attribute]['mean'] * 100 for llm in llm_names}
    
    # 找出最好和次好的LLM（对于每个指标）
    def get_best_second(values_dict, higher_is_better=True):
        sorted_llms = sorted(values_dict.keys(), key=lambda x: values_dict[x], reverse=higher_is_better)
        return sorted_llms[0] if len(sorted_llms) > 0 else None, sorted_llms[1] if len(sorted_llms) > 1 else None
    
    acc_best, acc_second = get_best_second(acc_values, higher_is_better=True)
    delta_acc_best, delta_acc_second = get_best_second(delta_acc_values, higher_is_better=False)
    dp_best, dp_second = get_best_second(dp_values, higher_is_better=False)
    eod_best, eod_second = get_best_second(eod_values, higher_is_better=False)
    fut_best, fut_second = get_best_second(fut_values, higher_is_better=True)
    
    # 格式化每个LLM的结果
    formatted_results = {}
    
    for llm in llm_names:
        summary = llm_results_dict[llm]
        
        # 提取所有值
        acc_mean = summary['ACC']['mean'] * 100
        acc_c_low = summary['ACC']['c_low'] * 100
        acc_c_high = summary['ACC']['c_high'] * 100
        
        delta_acc_mean = summary['Delta-ACC'][attribute]['mean'] * 100
        delta_acc_c_low = summary['Delta-ACC'][attribute]['c_low'] * 100
        delta_acc_c_high = summary['Delta-ACC'][attribute]['c_high'] * 100
        
        dp_mean = summary['DP'][attribute]['mean'] * 100
        dp_c_low = summary['DP'][attribute]['c_low'] * 100
        dp_c_high = summary['DP'][attribute]['c_high'] * 100
        
        eod_mean = summary['EoD'][attribute]['mean'] * 100
        eod_c_low = summary['EoD'][attribute]['c_low'] * 100
        eod_c_high = summary['EoD'][attribute]['c_high'] * 100
        
        fut_mean = summary['FUT'][attribute]['mean'] * 100
        fut_c_low = summary['FUT'][attribute]['c_low'] * 100
        fut_c_high = summary['FUT'][attribute]['c_high'] * 100
        
        # 格式化函数，根据是否最好/次好添加标记
        def format_value(mean, c_low, c_high, is_best, is_second):
            base = f"{mean:.2f}_{{[{c_low:.2f}, {c_high:.2f}]}}"
            if is_best:
                return f"\\xbest{{{base}}}"
            elif is_second:
                return f"\\xsecond{{{base}}}"
            else:
                return base
        
        # 生成LaTeX字符串
        latex_str = f"\\textbf{{{llm}}} & "
        latex_str += f"${format_value(acc_mean, acc_c_low, acc_c_high, llm == acc_best, llm == acc_second)}$ & "
        latex_str += f"${format_value(delta_acc_mean, delta_acc_c_low, delta_acc_c_high, llm == delta_acc_best, llm == delta_acc_second)}$ & "
        latex_str += f"${format_value(dp_mean, dp_c_low, dp_c_high, llm == dp_best, llm == dp_second)}$ & "
        latex_str += f"${format_value(eod_mean, eod_c_low, eod_c_high, llm == eod_best, llm == eod_second)}$ & "
        latex_str += f"${format_value(fut_mean, fut_c_low, fut_c_high, llm == fut_best, llm == fut_second)}$ \\\\\\\\"
        
        formatted_results[llm] = latex_str
    
    return formatted_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute fairness metrics from the given input csv.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Name of the dataset.")
    parser.add_argument("--llm", type=str, nargs='+', required=True,
                        help="Name(s) of the LLM(s). Can specify multiple LLMs separated by space.")
    args = parser.parse_args()

    # 如果提供了多个LLM，收集所有结果并使用标记最好/次好的格式
    if len(args.llm) > 1:
        print(f"\nProcessing {len(args.llm)} LLMs: {args.llm}")
        all_bootstrap_summaries = {}
        
        for llm in args.llm:
            print(f"\nProcessing {llm}...")
            data_dict = load_data(args.dataset, llm)
            bootstrap_summary, all_bootstrap = bootstrap_metrics(
                data_dict, n_iterations=1000, confidence_level=0.95)
            all_bootstrap_summaries[llm] = bootstrap_summary
        
        # 格式化所有结果，标记最好和次好的
        print("\nFormatting results with best/second-best marking...")
        formatted_gender_results = format_multiple_bootstrap_results(all_bootstrap_summaries, attribute='gender')
        formatted_age_results = format_multiple_bootstrap_results(all_bootstrap_summaries, attribute='age_raw')
        
        # 输出结果
        print("\n" + "="*70)
        print("Gender-based metrics:")
        print("="*70)
        for llm in args.llm:
            print(formatted_gender_results[llm])
        
        print("\n" + "="*70)
        print("Age-based metrics:")
        print("="*70)
        for llm in args.llm:
            print(formatted_age_results[llm])
        
        # 保存到文件
        with open(f"bootstrap_results_comparison.txt", "w") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Bootstrap Results for {args.dataset}\n")
            f.write("="*70 + "\n\n")
            f.write("Gender-based metrics:\n")
            f.write("-"*70 + "\n")
            for llm in args.llm:
                f.write(formatted_gender_results[llm] + "\n")
            f.write("\n" + "="*70 + "\n\n")
            f.write("Age-based metrics:\n")
            f.write("-"*70 + "\n")
            for llm in args.llm:
                f.write(formatted_age_results[llm] + "\n")
        
        print(f"\nResults saved to bootstrap_results_comparison.txt")
    
    else:
        # 单个LLM的原有逻辑
        llm = args.llm[0]
        data_dict = load_data(args.dataset, llm)

        # 原始指标
        metrics_dict = compute_fairness_metrics(data_dict)
        formatted_result = format_result(llm, metrics_dict)

        # Bootstrap 分析
        print("\nPerforming bootstrap analysis...")
        bootstrap_summary, all_bootstrap = bootstrap_metrics(
            data_dict, n_iterations=1000, confidence_level=0.95)
        latex_results = format_bootstrap_result(llm, bootstrap_summary)
        ic(latex_results)

        with open(f"bootstrap_results.txt", "a+") as f:
            f.write("\n" + "="*70 + "\n")
            f.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Bootstrap Results for {args.dataset} with LLM {llm}:\n")
            f.write(latex_results['gender'] + "\n")
            f.write(latex_results['age'] + "\n")
