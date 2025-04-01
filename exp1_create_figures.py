import os
import json
import fire
import logging
import matplotlib.pyplot as plt
import numpy as np
import nltk
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from focus_calculator import FocusCalculator
import textwrap
nltk.download('punkt')

logger = logging.getLogger("mallm")

NUM_TURNS = 7
GENERATIVE_TASKS = ["etpc", "xsum", "wmt19_de_en"]
REASONING_TASKS = ["strategyqa", "winogrande", "aqua_rat"]
KNOWLEDGE_TASKS = ["gpqa", "mmlu_pro", "ethics"]
TASKS_ORDER = GENERATIVE_TASKS + REASONING_TASKS + KNOWLEDGE_TASKS + ["ifeval"]


original_print = print

plt.rcParams.update({'font.size': 20})

with open("metrics_for_figures.json", "r") as metrics_file:
    metrics = json.load(metrics_file)

def format(text):
    text = text.replace("-public", "")
    text = text.capitalize().replace("Correct", "Accuracy").replace("Correct", "Accuracy").replace("Rougel", "ROUGE-L").replace("Rouge1", "ROUGE-1").replace("Rouge2", "ROUGE-2").replace("Bleu", "BLEU").replace("Distinct1", "Distinct-1").replace("Distinct2", "Distinct-2").replace("Bertscore", "BERTScore")
    text = text.replace("Simple_ethical_questions", "Ethical Questions").replace("Simple", "Ethical Questions").replace("Squad_v2", "SQuAD 2.0").replace("Wmt19_de_en", "WMT19 de-en").replace("Etpc", "ETPC").replace("Strategyqa", "StrategyQA").replace("Xsum", "XSum").replace("Squad", "SQuAD 2.0").replace("Winogrande", "WinoGrande").replace("Aqua_rat", "AQUA-RAT").replace("Gpqa", "GPQA").replace("Mmlu_pro", "MMLU-Pro").replace("Ifeval", "IFEval")
    return text

def corr_heatmap_with_pval(df, method = 'pearson', figsize=(14, 6), filename=None, title=None):
    """
    df: dataframe to be used. Ensured the dataframe has been sliced to contain only the column you need. It accepts only numerical columns
    method: default uses the pearson method. It overall permits 3 methods; 'pearson', 'spearman' and 'kendall'
    figsize: default is (20, 10) but you can change it based on your preference
    title: Specify the title for your chart, default is None
    """
    # Make a copy of the df
    data = df.copy()
   
    # Log initial size and NaN counts
    initial_size = len(data)
    nan_counts = data.isna().sum()
    logger.info(f"\nInitial data size: {initial_size} rows")
    logger.info("NaN counts per column:")
    for col, count in nan_counts.items():
        logger.info(f"  {col}: {count} NaN values ({(count/initial_size)*100:.1f}%)")
    data = data.dropna()
    final_size = len(data)
    dropped_rows = initial_size - final_size
    logger.info(f"Dropped {dropped_rows} rows ({(dropped_rows/initial_size)*100:.1f}% of data) containing NaN values")
    logger.info(f"Final data size: {final_size} rows")
    if final_size == 0:
        logger.warning(f"No valid data points remaining after dropping NaN values for {filename}")
        return

    # Check features correlation
    corr = data.corr(method = method)
    
    # Check if correlation matrix contains any valid values
    if corr.isnull().all().all():
       logger.warning(f"Correlation matrix is all NaN for {filename}")
       return
    
    # Remove the first row and last column from corr
    corr = corr.iloc[1:, :-1]

    # Create a mask to hide the upper triangle and element at position [0,1]
    mask = np.triu(np.ones_like(corr, dtype=bool))
    mask[0, 1] = True
    print(corr)

    # Set the diagonal elements of the mask to True to hide self-correlation
    np.fill_diagonal(mask, False)

    fig, ax = plt.subplots(figsize=figsize)
    
    plt.title(title, fontsize=50)
    #plt.tight_layout()


    sns.heatmap(corr,
                annot=True,
                annot_kws={"fontsize": 50},  # Adjust annotation font size
                fmt='.2f',
                linewidths=0.5,
                cmap=sns.diverging_palette(250, 15, s=75, l=40, center="dark", as_cmap=True),
                mask=mask,
                ax=ax,
                center=0,
                vmin = -1.0,
                vmax = 1.0,
                cbar_kws={"shrink": .9}
    )

    # Create a function to calculate and format p-values
    p_values = np.full((corr.shape[0], corr.shape[1]), np.nan)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            x = data.iloc[:, i].astype(float)
            y = data.iloc[:, j].astype(float)
            mask = ~np.logical_or(np.isnan(x), np.isnan(y))
            if np.sum(mask) > 0:
                if method == 'pearson':
                    p_values[i, j] = pearsonr(x[mask], y[mask])[1] #Changes based on the method chosen in the function
                elif method == 'kendall':
                    p_values[i, j] = kendalltau(x[mask], y[mask])[1]
                elif method == 'spearman':
                    p_values[i, j] = spearmanr(x[mask], y[mask])[1]

    p_values = pd.DataFrame(p_values, columns=corr.columns, index=corr.index)

    # Calculate the highest and lowest correlation coefficients
    np.max(corr.max())
    np.min(corr.min())

    # Annotate the heatmap with p-values and change text color based on correlation value
    for i in range(p_values.shape[0]):
        for j in range(p_values.shape[1]):
            p_value = p_values.iloc[i, j]
            if not np.isnan(p_value):
                corr.iloc[i, j]
                text_color = 'white'
                if p_value <= 0.001:
                    ax.text(i + 0.5, j + 0.8, '(p < 0.001)',
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=40,
                            color=text_color)
                elif p_value <= 0.01:
                    #include double asterisks for p-value <= 0.01
                    ax.text(i + 0.5, j + 0.8, f'(p = {p_value:.3f})',
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=40,
                            color=text_color)
                elif p_value <= 0.05:
                    #include single asterisks for p-value <= 0.05
                    ax.text(i + 0.5, j + 0.8, f'(p = {p_value:.3f})',
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=40,
                            color=text_color)
                else:
                    ax.text(i + 0.5, j + 0.8, f'(p = {p_value:.3f})',
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=40,
                            color=text_color)

    # Customize x-axis labels
    x_labels = [textwrap.fill(label.get_text(), 12) for label in ax.get_xticklabels()]
    ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=50)

    # Customize y-axis labels
    y_labels = [textwrap.fill(label.get_text(), 12) for label in ax.get_yticklabels()]
    ax.set_yticklabels(y_labels, rotation=0, ha="right", fontsize=50)
    ax.grid(False)
    ax.collections[0].set_clim(-0.5,0.5)  

    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cbar.set_ticklabels(['-0.5', '', '', '', '', '0', '', '', '', '', '0.5'])
    # Increase font size of colorbar ticks
    cbar.ax.tick_params(labelsize=50)
    
    output_file = os.path.join("exp1/figures", filename)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

def average_score_per_turn(eval_data, dataset, metric_name, baseline_data = None):
    scores_per_turn, thetas_per_turn, total_thetas, solutions_per_turn = [], [], [], []

    for sample in eval_data:
        scores, thetas, total_theta, solutions = FocusCalculator.calculate_per_turn(sample)
        thetas_per_turn.append(thetas)
        scores_per_turn.append(scores)
        solutions_per_turn.append(solutions)
        total_thetas.append(total_theta)

    positive_theta_indices = [i for i, theta in enumerate(total_thetas) if theta > 0]
    negative_theta_indices = [i for i, theta in enumerate(total_thetas) if theta < 0]
    all_theta_indices = [i for i in range(len(total_thetas))]
    
    positive_discussions = [eval_data[i] for i in positive_theta_indices]
    negative_discussions = [eval_data[i] for i in negative_theta_indices]
    all_discussions = [eval_data[i] for i in all_theta_indices]

    plt.figure(figsize=(10, 6))
    
    avg_scores_per_turn_positive = [0] * NUM_TURNS
    for turn in range(NUM_TURNS):
        scores = [discussion["votesEachTurn"][str(turn+1)]["alterations"]["public"]["score"][metrics[os.path.splitext(os.path.basename(sample["dataset"]))[0]][0]+"-public"] for discussion in positive_discussions if str(turn+1) in discussion["votesEachTurn"]]
        scores = [0 if score is None else score for score in scores]
        if scores:
            avg_scores_per_turn_positive[turn] = np.mean(scores)
    plt.plot(range(1, NUM_TURNS + 1), avg_scores_per_turn_positive, marker='o', linestyle='-', color='royalblue', label="Improving Discussions")
    plt.text(NUM_TURNS-0.2, avg_scores_per_turn_positive[-1], str(len(positive_discussions)) + " discussions", color='royalblue', ha='right')
    avg_scores_per_turn_negative = [0] * NUM_TURNS
    for turn in range(NUM_TURNS):
        scores = [discussion["votesEachTurn"][str(turn+1)]["alterations"]["public"]["score"][metrics[os.path.splitext(os.path.basename(sample["dataset"]))[0]][0]+"-public"] for discussion in negative_discussions if str(turn+1) in discussion["votesEachTurn"]]
        scores = [0 if score is None else score for score in scores]
        if scores:
            avg_scores_per_turn_negative[turn] = np.mean(scores)
    plt.plot(range(1, NUM_TURNS + 1), avg_scores_per_turn_negative, marker='o', linestyle='-', color='indianred', label="Worsening Discussions")
    plt.text(NUM_TURNS-0.2, avg_scores_per_turn_negative[-1], str(len(negative_discussions)) + " discussions", color='indianred', ha='right')
    avg_scores_per_turn_all = [0] * NUM_TURNS
    for turn in range(NUM_TURNS):
        scores = [discussion["votesEachTurn"][str(turn+1)]["alterations"]["public"]["score"][metrics[os.path.splitext(os.path.basename(sample["dataset"]))[0]][0]+"-public"] for discussion in all_discussions if str(turn+1) in discussion["votesEachTurn"]]
        scores = [0 if score is None else score for score in scores]
        if scores:
            avg_scores_per_turn_all[turn] = np.mean(scores)
    plt.plot(range(1, NUM_TURNS + 1), avg_scores_per_turn_all, marker='o', linestyle='-', color='grey', label="All Discussions")
    plt.text(NUM_TURNS-0.2, avg_scores_per_turn_all[-1], str(len(all_discussions)) + " discussions", color='grey', ha='right')
    print("avg_scores_per_turn_all: " + str(avg_scores_per_turn_all))
    print("avg_scores_per_turn_positive: " + str(avg_scores_per_turn_positive))
    print("avg_scores_per_turn_negative: " + str(avg_scores_per_turn_negative))

    if baseline_data:
        print("baseline data with none scores: " + str(len([sample for sample in baseline_data if sample["scores"][metrics[os.path.splitext(os.path.basename(sample["dataset"]))[0]][0]] is None])))
        average_baseline_score = np.mean([sample["scores"][metrics[os.path.splitext(os.path.basename(sample["dataset"]))[0]][0]] for sample in baseline_data if sample["scores"][metrics[os.path.splitext(os.path.basename(sample["dataset"]))[0]][0]] is not None])
        plt.axhline(average_baseline_score, color='black', linestyle='--', label='Single-Agent CoT')

    plt.xlabel('Turn')
    plt.ylabel(f'Average {metric_name}')
    plt.title(f'Performance per Turn for {format(dataset)}')
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.xticks(range(1, NUM_TURNS + 1))
    plt.ylim(-0.05, 1.05)
    plt.savefig(os.path.join("exp1/figures", f"average_score_per_turn_{dataset}.pdf"))
    plt.close()

def mallm_vs_baseline():
    stats_baseline, _, _ = get_experiment_stats_and_eval_data(baseline = True)
    stats, _, _ = get_experiment_stats_and_eval_data(baseline = False)

    plt.figure(figsize=(10, 6))
    
    mallm_scores = []
    baseline_scores = []
    datasets = []
    error_bars_mallm = []
    error_bars_baseline = []
    avg_scores_per_turn1 = []
    avg_scores_per_turn7 = []
    std_dev_scores_per_turn1 = []
    std_dev_scores_per_turn7 = []
    

    for dataset in TASKS_ORDER:
        if dataset not in stats or dataset not in stats_baseline:
            continue
            
        metric = metrics[dataset][0]
        datasets.append(format(dataset))
        mallm_scores.append(stats[dataset][metric]["average_score"])
        baseline_scores.append(stats_baseline[dataset][metric]["average_score"])
        error_bars_mallm.append(stats[dataset][metric]["std_dev_score"])
        error_bars_baseline.append(stats_baseline[dataset][metric]["std_dev_score"])

        avg_scores_per_turn1.append(stats[dataset][metric]["avg_scores_per_turn"][0])
        avg_scores_per_turn7.append(stats[dataset][metric]["avg_scores_per_turn"][6])
        std_dev_scores_per_turn1.append(stats[dataset][metric]["std_dev_scores_per_turn"][0])
        std_dev_scores_per_turn7.append(stats[dataset][metric]["std_dev_scores_per_turn"][6])
        
    # Plot bars
    x = np.arange(len(datasets))
    width = 0.25
    # Plot turn 2 and turn 7 scores
    plt.bar(x - width, baseline_scores, width, label='Single-Agent CoT', color='grey', yerr=error_bars_baseline, alpha=0.7, capsize=4)
    plt.bar(x, avg_scores_per_turn1, width, label='MALLM Turn 1', color='lightgreen', yerr=std_dev_scores_per_turn1, alpha=0.7, capsize=4)
    plt.bar(x + width, avg_scores_per_turn7, width, label='MALLM Turn 7', color='darkgreen', yerr=std_dev_scores_per_turn7, alpha=0.7, capsize=4)

    plt.xlabel('Dataset')
    plt.ylabel('Average Performance')
    plt.title('MALLM vs Single-Agent CoT Performance')
    plt.xticks(x, datasets, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    plt.savefig(os.path.join("exp1/figures", "mallm_vs_baseline.pdf"))
    plt.close()

def corr_persona_diversity_problem_focus(eval_data_all):
    eval_data_all = [item for sublist in eval_data_all for item in sublist]
    persona_diversities = []
    problem_focus_scores = []

    for sample in eval_data_all:
        persona_diversities.append(sample["persona_diversity"])
        scores, thetas, total_theta, solutions = FocusCalculator.calculate_per_turn(sample)
        problem_focus_scores.append(total_theta)

    plt.scatter(persona_diversities, problem_focus_scores)
    plt.xlim(0, 1)
    plt.ylim(-1, 1)
    z = np.polyfit(persona_diversities, problem_focus_scores, 1)
    p = np.poly1d(z)
    plt.plot(persona_diversities, p(persona_diversities), "r--")
    plt.xlabel("Persona Diversity")
    plt.ylabel("Problem Focus")
    plt.title("Correlation between Persona Diversity and Problem Focus Score")
    plt.savefig(os.path.join("exp1/figures", "corr_persona_diversity_problem_focus.pdf"))
    plt.close()

def successful_samples():
    _, _, eval_data_all = get_experiment_stats_and_eval_data(baseline = False)  
    #tasks = [task for task in TASKS_ORDER if task not in GENERATIVE_TASKS]
    tasks = TASKS_ORDER
    eval_data_all = {k: v for k, v in eval_data_all.items() if k in tasks}
    fig, ax = plt.subplots()
    ax.set_ylabel('% of Total Samples', fontsize=15)
    x = np.arange(len(tasks))
    ax.set_xlim(-0.5, len(tasks))
    ax.set_xticks(x)
    width = 0.35
    datasets = []
    all_turns_to_recover = [[], [], []]
    all_drift_strengths = [[], [], []]
    all_drift_strengths_recovered = [[], [], []]
    all_recovered_performance_differences = [[], [], []]

 
    for k, dataset in enumerate(tasks):

        drifting_samples_indices = [set(), set(), set()]
        recovering_indices = [[], [], []]
        turns_to_recover = [[], [], []]
        recovered_performance_differences = [[], [], []]
        drift_strengths = [[], [], []]
        drift_strengths_recovered = [[], [], []]
        datasets.append(format(dataset))
        
        for j in range(3):
            eval_data = eval_data_all[dataset][j]

            scores_per_turn, thetas_per_turn, total_thetas, solutions_per_turn = [], [], [], []

            for index, sample in enumerate(eval_data):
                scores, thetas, total_theta, solutions = FocusCalculator.calculate_per_turn(sample)
                thetas_per_turn.append(thetas)
                scores_per_turn.append(scores)
                solutions_per_turn.append(solutions)
                total_thetas.append(total_theta)

                highest_score = 0
                past_score = 1
                recent_score_before_recovery = 1
                recent_turn_before_recovery = None
                drift_strength = 0
                for i, score in enumerate(scores):
                    if score > highest_score:
                        # update high
                        highest_score = score
                    if score < past_score and i != 0:
                        # problem drift
                        recent_score_before_recovery = score
                        recent_turn_before_recovery = i
                        drift_strength = highest_score - score
                        drifting_samples_indices[j].add(index)
                    if score > recent_score_before_recovery and score >= highest_score:
                        if not recent_turn_before_recovery:
                            print("WARNING: Recent turn before recovery is None")
                            recent_turn_before_recovery = 1
                        # recovered
                        recovering_indices[j].append(index)
                        turns_to_recover[j].append(i-recent_turn_before_recovery)
                        all_turns_to_recover[j].append(i-recent_turn_before_recovery)
                        recovered_performance_differences[j].append(score-recent_score_before_recovery)
                        all_recovered_performance_differences[j].append(score-recent_score_before_recovery)
                        drift_strengths_recovered[j].append(drift_strength)
                        all_drift_strengths_recovered[j].append(drift_strength)
                        break
                    past_score = score
                if drift_strength > 0:
                    drift_strengths[j].append(drift_strength)
                    all_drift_strengths[j].append(drift_strength)
        
        print(f"--> {k}: " + dataset)
        total_samples = np.mean([len(eval_data_all[dataset][i]) for i in range(3)])
        total_samples_std_dev = np.std([len(eval_data_all[dataset][i]) for i in range(3)])
        print(f"Total samples: {total_samples}, Std-Dev: {total_samples_std_dev}")

        drifting_samples = np.mean([len(list(drifting_samples_indices[i])) for i in range(3)])
        drifting_samples_std_dev = np.std([len(list(drifting_samples_indices[i])) for i in range(3)])
        drifting_samples_percentage = drifting_samples/total_samples*100    
        drifting_samples_percentage_std_dev = np.std([len(list(drifting_samples_indices[i]))/total_samples*100 for i in range(3)])
        print(f"Drifting: {drifting_samples}, Std-Dev: {drifting_samples_std_dev}, {drifting_samples_percentage:.2f}% of total samples (+-{drifting_samples_percentage_std_dev:.2f}%)")

        successful_samples = np.mean([len(eval_data_all[dataset][i]) - len(list(drifting_samples_indices[i])) for i in range(3)])
        successful_samples_std_dev = np.std([len(eval_data_all[dataset][i]) - len(list(drifting_samples_indices[i])) for i in range(3)])
        successful_samples_percentage = successful_samples/total_samples*100
        successful_samples_percentage_std_dev = np.std([(len(eval_data_all[dataset][i]) - len(list(drifting_samples_indices[i])))/total_samples*100 for i in range(3)])
        print(f"Successful: {successful_samples}, Std-Dev: {successful_samples_std_dev}, {successful_samples_percentage:.2f}% of total samples")
        
        recovering_samples = np.mean([len(recovering_indices[i]) for i in range(3)])
        recovering_samples_std_dev = np.std([len(recovering_indices[i]) for i in range(3)])
        recovering_samples_percentage = recovering_samples/total_samples*100
        recovering_samples_percentage_std_dev = np.std([len(recovering_indices[i])/total_samples*100 for i in range(3)])
        print(f"Recovering: {recovering_samples}, Std-Dev: {recovering_samples_std_dev}, {recovering_samples_percentage:.2f}% of total samples (+-{recovering_samples_percentage_std_dev:.2f}%)")

        recovering_samples_from_drifting_percentage = recovering_samples/drifting_samples*100
        recovering_samples_from_drifting_percentage_std_dev = np.std([len(recovering_indices[i])/len(drifting_samples_indices[i])*100 for i in range(3)])
        print(f"Recovering from Drifting: {recovering_samples}, Std-Dev: {recovering_samples_std_dev}, {recovering_samples_from_drifting_percentage:.2f}% of drifting samples (+-{recovering_samples_from_drifting_percentage_std_dev:.2f}%)")

        print(f"All good samples: {successful_samples+recovering_samples}, Std-Dev: {np.std([successful_samples+recovering_samples for i in range(3)])}, {(successful_samples+recovering_samples)/total_samples*100:.2f}% of total samples")
        
        print(f"Avg. number of turns to recover: {np.mean([np.mean(turns_to_recover[i]) for i in range(3)])} turns, Std-Dev: {np.std([np.mean(turns_to_recover[i]) for i in range(3)])}")
        print(f"Avg. drift strength: {np.mean([np.mean(drift_strengths[i]) for i in range(3)])}, Std-Dev: {np.std([np.mean(drift_strengths[i]) for i in range(3)])}")
        print(f"Avg. drift strength recovered from: {np.mean([np.mean(drift_strengths_recovered[i]) for i in range(3)])}, Std-Dev: {np.std([np.mean(drift_strengths_recovered[i]) for i in range(3)])}")
        print(f"Avg. Performance difference (low-new_high): {np.mean([np.mean(recovered_performance_differences[i]) for i in range(3)])}, Std-Dev: {np.std([np.mean(recovered_performance_differences[i]) for i in range(3)])}")

        #rects1 = ax.bar(x[k] - width/3, total_samples, width, label='Total Samples', color='grey')
        rects2 = ax.bar(x[k], successful_samples_percentage, width, yerr=successful_samples_percentage_std_dev, label='Never Drifted', color='royalblue', capsize=3, alpha=0.7)
        rects3 = ax.bar(x[k], recovering_samples_percentage, width, yerr=recovering_samples_percentage_std_dev, label='Recovered from Drift', color='seagreen', capsize=3, alpha=0.7, bottom=successful_samples_percentage)

        if k == 2 or k == 5 or k == 8:
            ax.axvline(x=x[k] + width/2 + 0.3, color='black', linestyle='--', alpha=0.3)
    
        # Add value labels on top of each bar
        for rect in rects2:
            height1 = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2. + 0.4, height1,
                    f'{height1:.1f}%',
                    ha='center', va='bottom', fontsize=10)
        
        for rect in rects3:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2. + 0.4, height1 + height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10)

    # unique labels
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(dict.fromkeys(labels))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    ax.legend(unique_handles, unique_labels, fontsize=15, loc='lower right')

    ax.set_xticklabels(datasets)
    plt.xticks(rotation=45, ha='right', fontsize=15)
    plt.yticks(fontsize=15)
    ax.set_ylim(70, 100)
    ax.set_yticks([70, 80, 90, 100])
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    ax.set_title('Successful Samples', fontsize=15)
    plt.savefig(os.path.join("exp1/figures", "successful_samples_all_tasks.pdf"))
    plt.close()

def calculate_avg_turning_points(eval_data):
    turning_points = []
    for sample in eval_data:
        _, thetas, _, _ = FocusCalculator.calculate_per_turn(sample)
        turning_points.append(len([theta for theta in thetas if theta < 0]))

    return np.mean(turning_points)

def get_experiment_stats_and_eval_data(baseline = False, dataset_to_process = None):
    
    datasets = TASKS_ORDER
    if dataset_to_process:
        datasets = [dataset_to_process]

    stats = {dataset: {} for dataset in datasets}
    eval_data = {dataset: {} for dataset in datasets}
    eval_data_seperated = {dataset: [] for dataset in datasets}
    baseline_str = ""
    if baseline:
        baseline_str = "_baseline"

    for dataset in datasets:
        with open(f"exp1/out/output_{dataset}{baseline_str}_repeat1-stats.json", "r") as f:
            stats1 = json.load(f) 
        with open(f"exp1/out/output_{dataset}{baseline_str}_repeat2-stats.json", "r") as f:
            stats2 = json.load(f) 
        with open(f"exp1/out/output_{dataset}{baseline_str}_repeat3-stats.json", "r") as f:
            stats3 = json.load(f) 

        turning_points1, turning_points2, turning_points3 = 0, 0, 0

        with open(f"exp1/out/output_{dataset}{baseline_str}_repeat1-eval.json", "r") as f:
            eval_data1 = json.load(f) 
            if not baseline:
                turning_points1 = calculate_avg_turning_points(eval_data1)
                avg_scores_per_turn1 = [0] * NUM_TURNS
                for turn in range(NUM_TURNS):
                    scores = [discussion["votesEachTurn"][str(turn+1)]["alterations"]["public"]["score"][metrics[dataset][0]+"-public"] for discussion in eval_data1]
                    scores = [0 if score is None else score for score in scores]
                    if scores:
                        avg_scores_per_turn1[turn] = np.mean(scores)
        with open(f"exp1/out/output_{dataset}{baseline_str}_repeat2-eval.json", "r") as f:
            eval_data2 = json.load(f) 
            if not baseline:
                turning_points2 = calculate_avg_turning_points(eval_data2)
                avg_scores_per_turn2 = [0] * NUM_TURNS
                for turn in range(NUM_TURNS):
                    scores = [discussion["votesEachTurn"][str(turn+1)]["alterations"]["public"]["score"][metrics[dataset][0]+"-public"] for discussion in eval_data2]
                    scores = [0 if score is None else score for score in scores]
                    if scores:
                        avg_scores_per_turn2[turn] = np.mean(scores)
        with open(f"exp1/out/output_{dataset}{baseline_str}_repeat3-eval.json", "r") as f:
            eval_data3 = json.load(f) 
            if not baseline:
                turning_points3 = calculate_avg_turning_points(eval_data3)
                avg_scores_per_turn3 = [0] * NUM_TURNS
                for turn in range(NUM_TURNS):
                    scores = [discussion["votesEachTurn"][str(turn+1)]["alterations"]["public"]["score"][metrics[dataset][0]+"-public"] for discussion in eval_data3]
                    scores = [0 if score is None else score for score in scores]
                    if scores:
                        avg_scores_per_turn3[turn] = np.mean(scores)

        eval_data[dataset] = eval_data1 + eval_data2 + eval_data3
        eval_data_seperated[dataset] = [eval_data1, eval_data2, eval_data3]
        
        avg_scores_per_turn = [0] * NUM_TURNS
        std_dev_per_turn = [0] * NUM_TURNS
        if not baseline:
            for turn in range(NUM_TURNS):
                # Get scores for each run at this turn
                scores1 = [discussion["votesEachTurn"][str(turn+1)]["alterations"]["public"]["score"][metrics[dataset][0]+"-public"] for discussion in eval_data1]
                scores2 = [discussion["votesEachTurn"][str(turn+1)]["alterations"]["public"]["score"][metrics[dataset][0]+"-public"] for discussion in eval_data2]
                scores3 = [discussion["votesEachTurn"][str(turn+1)]["alterations"]["public"]["score"][metrics[dataset][0]+"-public"] for discussion in eval_data3]
                
                # Replace None with 0
                scores1 = [0 if score is None else score for score in scores1]
                scores2 = [0 if score is None else score for score in scores2]
                scores3 = [0 if score is None else score for score in scores3]
                
                # Calculate mean for each run
                mean1 = np.mean(scores1) if scores1 else 0
                mean2 = np.mean(scores2) if scores2 else 0
                mean3 = np.mean(scores3) if scores3 else 0
                
                # Calculate overall mean and std dev across runs
                avg_scores_per_turn[turn] = np.mean([mean1, mean2, mean3])
                std_dev_per_turn[turn] = np.std([mean1, mean2, mean3])

        overall_stats = {} 
        for metric in stats1:
            avg = (stats1[metric]["averageScore"] + stats2[metric]["averageScore"] + stats3[metric]["averageScore"]) / 3
            overall_stats[metric] = {
                "average_score": avg,
                "std_dev_score": np.std([stats1[metric]["averageScore"], stats2[metric]["averageScore"], stats3[metric]["averageScore"]]),
                "scores": stats1[metric]["scores"],
                "avg_turning_points": (turning_points1 + turning_points2 + turning_points3) / 3,
                "std_dev_turning_points": np.std([turning_points1, turning_points2, turning_points3]),
                "avg_scores_per_turn": avg_scores_per_turn,
                "std_dev_scores_per_turn": std_dev_per_turn
            }

        stats[dataset] = overall_stats

    output_path = "exp1/figures/_eval_stats.json"
    if baseline:
        output_path = "exp1/figures/_eval_stats_baseline.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=4)

    return stats, eval_data, eval_data_seperated


def get_percentage_of_ties(eval_data):
    ties = 0
    agreements = 0
    for sample in eval_data:
        for turn in range(NUM_TURNS):
            if sample["votesEachTurn"][str(turn+1)]["alterations"]["public"]["agreed"] is False:
                ties += 1
            else:
                agreements += 1
    print(f"Agreements: {agreements}, Ties: {ties}, {agreements/(agreements+ties)*100:.2f}% of total samples")


def main():
    eval_data_all = []
    baseline_data_all = []
    baseline_data_mulchoice = []
    eval_data_mulchoice = []
    for dataset in TASKS_ORDER:
        print("-> Processing baseline data for " + dataset)
        stats, baseline_data, _ = get_experiment_stats_and_eval_data(baseline = True, dataset_to_process = dataset)
        baseline_data = baseline_data[dataset]
        print("-> Samples: " + str(len(baseline_data)))
        baseline_data_all.append(baseline_data)
        if dataset in REASONING_TASKS+KNOWLEDGE_TASKS:
            baseline_data_mulchoice.extend(baseline_data)
        
        for metric in stats[dataset]:
            print(f"Average score for {metric}: {round(stats[dataset][metric]['average_score'], 4)}")

    for dataset in TASKS_ORDER:
        print("-> Processing MALLM data for " + dataset)
        stats, eval_data, _ = get_experiment_stats_and_eval_data(baseline = False, dataset_to_process = dataset)
        eval_data = eval_data[dataset]
        print("-> Samples: " + str(len(eval_data)))
        eval_data_all.append(eval_data)
        if dataset in REASONING_TASKS+KNOWLEDGE_TASKS:
            eval_data_mulchoice.extend(eval_data)
        for metric in stats[dataset]:
            print(f"Average score for {metric}: {round(stats[dataset][metric]['average_score'], 4)}")
        
        # CORRELATION HEATMAP
        response_length = []
        solution_length = []
        agreeing_agents = []
        for s in eval_data:
            for i, m in enumerate(s["globalMemory"]):
                if m["agreement"] is False:
                    response_length.append(len(m["message"].split()))
                    solution_length.append(len(m["solution"].split()))
                    agents_that_agree = 0
                    if i+1 < len(s["globalMemory"]) and s["globalMemory"][i+1]["agreement"]:
                        agents_that_agree += 1
                        if i+2 < len(s["globalMemory"]) and s["globalMemory"][i+2]["agreement"]:
                            agents_that_agree +=1
                    agreeing_agents.append(agents_that_agree)
        combined_data = pd.DataFrame(
            {
                'Response Length': response_length, 
                'Solution Length': solution_length, 
                'Agreeing Agents': agreeing_agents
            }
        )


        metric_name = format(metrics[dataset][0])
        corr_heatmap_with_pval(combined_data, method = 'spearman', filename=f"{dataset}_lenAgree_OnlyDrafts_corr.pdf", title=f"Correlations for {format(dataset)}")
        average_score_per_turn(eval_data, dataset, metric_name, None)  # baseline_data

    # CORRELATION HEATMAP
    response_length = []
    solution_length = []
    agreeing_agents = []
    for s in eval_data_mulchoice:
        for i, m in enumerate(s["globalMemory"]):
            if m["agreement"] is False:
                response_length.append(len(m["message"].split()))
                solution_length.append(len(m["solution"].split()))
                agents_that_agree = 0
                if i+1 < len(s["globalMemory"]) and s["globalMemory"][i+1]["agreement"]:
                    agents_that_agree += 1
                    if i+2 < len(s["globalMemory"]) and s["globalMemory"][i+2]["agreement"]:
                        agents_that_agree +=1
                agreeing_agents.append(agents_that_agree)
    combined_data = pd.DataFrame(
        {
            'Response Length': response_length, 
            'Solution Length': solution_length, 
            'Agreeing Agents': agreeing_agents
        }
    )
    print(f"Number of samples in multiple-choice correlation analysis: {len(combined_data)}")

    mallm_vs_baseline()
    average_score_per_turn(eval_data_mulchoice, "Multiple-Choice Datasets", "Accuracy", None) # baseline_data_mulchoice
    corr_heatmap_with_pval(combined_data, method = 'spearman', filename="all_lenAgree_OnlyDrafts_corr.pdf", title="Correlations for Multiple-Choice Datasets")
    corr_persona_diversity_problem_focus(eval_data_all)
    get_percentage_of_ties([item for sublist in eval_data_all for item in sublist])
    successful_samples()

if __name__ == "__main__":
    fire.Fire(main)