import os
import json
import fire
import logging
import matplotlib.pyplot as plt
import numpy as np
import nltk
from focus_calculator import FocusCalculator
nltk.download('punkt')

logger = logging.getLogger("mallm")

NUM_TURNS = 7
TASKS_ORDER = ["mmlu_pro"]
FOCUS_CALCULATOR = FocusCalculator()
FOCUS_CALCULATOR_FUNCTION = FOCUS_CALCULATOR.calculate_per_turn_embedding_similarity

original_print = print

plt.rcParams.update({'font.size': 20})

with open("metrics_for_figures.json", "r") as metrics_file:
    metrics = json.load(metrics_file)

def format(text):
    text = text.replace("-public", "")
    text = text.capitalize().replace("Correct", "Accuracy").replace("Correct", "Accuracy").replace("Rougel", "ROUGE-L").replace("Rouge1", "ROUGE-1").replace("Rouge2", "ROUGE-2").replace("Bleu", "BLEU").replace("Distinct1", "Distinct-1").replace("Distinct2", "Distinct-2")
    text = text.replace("Simple_ethical_questions", "Ethical Questions").replace("Simple", "Ethical Questions").replace("Squad_v2", "SQuAD 2.0").replace("Wmt19_de_en", "WMT19 de-en").replace("Etpc", "ETPC").replace("Strategyqa", "StrategyQA").replace("Xsum", "XSum").replace("Squad", "SQuAD 2.0").replace("Winogrande", "WinoGrande").replace("Aqua_rat", "AQUA-RAT").replace("Gpqa", "GPQA").replace("Mmlu_pro", "MMLU-Pro").replace("Ifeval", "IFEval")
    return text

def print_overall_stats(stats_normal, stats_policy, stats_regenerate, stats_policy_judge, stats_regenerate_judge):
    for dataset in TASKS_ORDER:
        metric = metrics[dataset][0]
        print(format(dataset))
        print("Normal --------------------------------")
        print("Average Score: " + str(stats_normal[dataset][metric]["average_score"]))
        print("Std Dev Score: " + str(stats_normal[dataset][metric]["std_dev_score"]))
        print("Policy --------------------------------")
        print("Average Score: " + str(stats_policy[dataset][metric]["average_score"]))
        print("Std Dev Score: " + str(stats_policy[dataset][metric]["std_dev_score"]))
        print("Regenerate --------------------------------")
        print("Average Score: " + str(stats_regenerate[dataset][metric]["average_score"]))
        print("Std Dev Score: " + str(stats_regenerate[dataset][metric]["std_dev_score"]))
        print("Policy Judge --------------------------------")
        print("Average Score: " + str(stats_policy_judge[dataset][metric]["average_score"]))
        print("Std Dev Score: " + str(stats_policy_judge[dataset][metric]["std_dev_score"]))
        print("Regenerate Judge --------------------------------")
        print("Average Score: " + str(stats_regenerate_judge[dataset][metric]["average_score"]))
        print("Std Dev Score: " + str(stats_regenerate_judge[dataset][metric]["std_dev_score"]))

def successful_samples(eval_data_normal, eval_data_policy, eval_data_regenerate, eval_data_policy_judge, eval_data_regenerate_judge):
    fig, ax = plt.subplots()
    ax.set_ylabel('% of 373 Samples', fontsize=15)
    x = np.arange(5)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(80, 100)
    ax.set_xticks(x)
    width = 0.35
    all_turns_to_recover = [[], [], []]
    all_drift_strengths = [[], [], []]
    all_drift_strengths_recovered = [[], [], []]
    all_recovered_performance_differences = [[], [], []]

    print(len(eval_data_policy))
    eval_data_methods = [eval_data_normal, eval_data_policy_judge, eval_data_regenerate_judge, eval_data_policy, eval_data_regenerate]
    dataset = "mmlu_pro"
 
    for k, eval_data_triple in enumerate(eval_data_methods):
        drifting_samples_indices = [set(), set(), set()]
        recovering_indices = [[], [], []]
        turns_to_recover = [[], [], []]
        recovered_performance_differences = [[], [], []]
        drift_strengths = [[], [], []]
        drift_strengths_recovered = [[], [], []]
        for j in range(3):
            eval_data = eval_data_triple[dataset][j]
            scores_per_turn, thetas_per_turn, total_thetas, solutions_per_turn = [], [], [], []

            for index, sample in enumerate(eval_data):
                scores, thetas, total_theta, solutions = FOCUS_CALCULATOR_FUNCTION(sample)
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
        total_samples = np.mean([len(eval_data_triple[dataset][i]) for i in range(3)])
        total_samples_std_dev = np.std([len(eval_data_triple[dataset][i]) for i in range(3)])
        print(f"Total samples: {total_samples}, Std-Dev: {total_samples_std_dev}")

        drifting_samples = np.mean([len(list(drifting_samples_indices[i])) for i in range(3)])
        drifting_samples_std_dev = np.std([len(list(drifting_samples_indices[i])) for i in range(3)])
        print(f"Drifting: {drifting_samples}, Std-Dev: {drifting_samples_std_dev}, {drifting_samples/total_samples*100:.2f}% of total samples")

        successful_samples = np.mean([len(eval_data_triple[dataset][i]) - len(list(drifting_samples_indices[i])) for i in range(3)])
        successful_samples_std_dev = np.std([len(eval_data_triple[dataset][i]) - len(list(drifting_samples_indices[i])) for i in range(3)])
        successful_samples_percentage = successful_samples/total_samples*100
        successful_samples_percentage_std_dev = np.std([(len(eval_data_triple[dataset][i]) - len(list(drifting_samples_indices[i])))/total_samples*100 for i in range(3)])
        print(f"Successful: {successful_samples}, Std-Dev: {successful_samples_std_dev}, {successful_samples_percentage:.2f}% of total samples")
        
        recovering_samples = np.mean([len(recovering_indices[i]) for i in range(3)])
        recovering_samples_std_dev = np.std([len(recovering_indices[i]) for i in range(3)])
        recovering_samples_percentage = recovering_samples/total_samples*100
        recovering_samples_percentage_std_dev = np.std([len(recovering_indices[i])/total_samples*100 for i in range(3)])
        print(f"Recovering: {recovering_samples}, Std-Dev: {recovering_samples_std_dev}, {recovering_samples_percentage:.2f}% of total samples")

        print(f"All good samples: {successful_samples+recovering_samples}, Std-Dev: {np.std([successful_samples+recovering_samples for i in range(3)])}, {(successful_samples+recovering_samples)/total_samples*100:.2f}% of total samples")
        
        print(f"Avg. number of turns to recover: {np.mean([np.mean(turns_to_recover[i]) for i in range(3)])} turns, Std-Dev: {np.std([np.mean(turns_to_recover[i]) for i in range(3)])}")
        print(f"Avg. drift strength: {np.mean([np.mean(drift_strengths[i]) for i in range(3)])}, Std-Dev: {np.std([np.mean(drift_strengths[i]) for i in range(3)])}")
        print(f"Avg. drift strength recovered from: {np.mean([np.mean(drift_strengths_recovered[i]) for i in range(3)])}, Std-Dev: {np.std([np.mean(drift_strengths_recovered[i]) for i in range(3)])}")
        print(f"Avg. Performance difference (low-new_high): {np.mean([np.mean(recovered_performance_differences[i]) for i in range(3)])}, Std-Dev: {np.std([np.mean(recovered_performance_differences[i]) for i in range(3)])}")

        #rects1 = ax.bar(x[k] - width/3, total_samples, width, label='Total Samples', color='grey')
        rects2 = ax.bar(x[k], successful_samples_percentage, width, yerr=successful_samples_percentage_std_dev, label='Never Drifted', color='royalblue', capsize=3, alpha=0.7)
        rects3 = ax.bar(x[k], recovering_samples_percentage, width, yerr=recovering_samples_percentage_std_dev, label='Recovered from Drift', color='seagreen', capsize=3, alpha=0.7, bottom=successful_samples_percentage)

        if k == 0 or k == 2:
            ax.axvline(x=x[k] + width/2 + 0.3, color='black', linestyle='--', alpha=0.3)
    
        # Add value labels on top of each bar
        for rect in rects2:
            height1 = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2. + 0.35, height1,
                    f'{height1:.1f}%',
                    ha='center', va='bottom', fontsize=12)
        
        for rect in rects3:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2. + 0.35, height1 + height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=12)

    # unique labels
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(dict.fromkeys(labels))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    ax.legend(unique_handles, unique_labels, fontsize=15, loc='lower right')

    ax.set_xticklabels(["Multi-Agent\nBaseline", "Policy\n(Judge)", "Regenerate\n(Judge)", "Policy\n(Oracle)", "Regenerate\n(Oracle)"])
    plt.xticks(rotation=45, ha='right', fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    ax.set_title('Successful Samples for MMLU-Pro', fontsize=15)
    plt.savefig(os.path.join("exp3/figures", "successful_samples.pdf"))
    plt.close()

def calculate_avg_turning_points(eval_data):
    turning_points = []
    for sample in eval_data:
        _, thetas, _, _ = FOCUS_CALCULATOR_FUNCTION(sample)
        turning_points.append(len([theta for theta in thetas if theta != 0]))

    return np.mean(turning_points)

def mallm_vs_baseline(stats):

    plt.figure(figsize=(10, 6))
    
    datasets = []
    avg_score = []
    std_dev_score = []
    avg_scores_per_turn1 = []
    avg_scores_per_turn7 = []
    std_dev_scores_per_turn1 = []
    std_dev_scores_per_turn7 = []
    
    methods = ["Single", "Normal", "Policy Judge", "Regenerate Judge"]

    for i, dataset in enumerate(methods):
            
        metric = metrics["mmlu_pro"][0]
        datasets.append(format(dataset))
        avg_score.append(stats[i]["mmlu_pro"][metric]["average_score"])
        std_dev_score.append(stats[i]["mmlu_pro"][metric]["std_dev_score"])
        avg_scores_per_turn1.append(stats[i]["mmlu_pro"][metric]["avg_scores_per_turn"][0])
        avg_scores_per_turn7.append(stats[i]["mmlu_pro"][metric]["avg_scores_per_turn"][6])
        std_dev_scores_per_turn1.append(stats[i]["mmlu_pro"][metric]["std_dev_scores_per_turn"][0])
        std_dev_scores_per_turn7.append(stats[i]["mmlu_pro"][metric]["std_dev_scores_per_turn"][6])
    
        if not dataset == "Single":
            print("----- Scores delta turn 7 - turn1")
            print(dataset)
            print("delta: " + str(round(stats[i]["mmlu_pro"][metric]["avg_scores_per_turn_delta"], 4)))
            print("std_dev: " + str(round(stats[i]["mmlu_pro"][metric]["avg_scores_per_turn_delta_std_dev"], 4)))

    # Plot bars
    width = 0.25
    x = np.arange(0,len(methods))
    # Plot turn 2 and turn 7 scores
    plt.bar(x-width/2, avg_scores_per_turn1, width, label='Debate Turn 1', color='lightgreen', yerr=std_dev_scores_per_turn1, alpha=0.7, capsize=4)
    plt.bar(x + width/2, avg_scores_per_turn7, width, label='Debate Turn 7', color='darkgreen', yerr=std_dev_scores_per_turn7, alpha=0.7, capsize=4)

    plt.bar(0, avg_score[0], width, label='CoT', color='grey', yerr=std_dev_score[0], alpha=0.7, capsize=4)

    plt.ylim(0.3, 0.7)
    for i in range(len(x)):
        if i == 0 or i == 1:
            plt.axvline(x=x[i] + 0.5, color='black', linestyle='--', alpha=0.3)
        if i == 0:
            plt.text(x[i] + 0.02 + width/2, avg_score[i] + 0.005, f'{round(avg_score[i], 4)}', ha='center', va='bottom', fontsize=12, color='black')
        else:
            plt.text(x[i] + 0.02, avg_scores_per_turn1[i] + 0.005, f'{round(avg_scores_per_turn1[i], 4)}', ha='center', va='bottom', fontsize=12, color='black')
            plt.text(x[i] + width + 0.02, avg_scores_per_turn7[i] + 0.005, f'{round(avg_scores_per_turn7[i], 4)}', ha='center', va='bottom', fontsize=12, color='black')
    plt.ylabel('Average Accuracy')
    plt.title('Multi-Agent Performance')
    plt.xticks(x, ["CoT\n(Baseline)", "Multi-Agent\n(Baseline)", "DRIFTJudge\n+DRIFTPolicy", "DRIFTJudge\n+Regenerate"], rotation=0, ha='center')
    plt.legend(loc='upper left')
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    plt.savefig(os.path.join("exp3/figures", "mallm_vs_baseline.pdf"))
    plt.close()

def get_eval_stats_and_eval_data(dataset_to_process = None):
    datasets = TASKS_ORDER
    if dataset_to_process:
        datasets = [dataset_to_process]

    stats = {dataset: {} for dataset in datasets}
    {dataset: {} for dataset in datasets}

    for dataset in datasets:
        with open(f"exp1/out/output_{dataset}_repeat1-stats.json", "r") as f:
            stats1 = json.load(f) 
            stats[dataset] = stats1


def get_experiment_stats_and_eval_data(llm_judge = False, intervention = "policy", dataset_to_process = None, baseline = False):
    
    datasets = TASKS_ORDER
    if dataset_to_process:
        datasets = [dataset_to_process]

    exp_dir = "exp3"
    stats = {dataset: {} for dataset in datasets}
    eval_data = {dataset: {} for dataset in datasets}
    eval_data_seperated = {dataset: [] for dataset in datasets}
    num_agents = 3
    llm_judge_str = ""
    if llm_judge:
        llm_judge_str = "_llmJudge"
    if intervention:
        intervention_str = "_" + intervention
        if "policy" in intervention:
            num_agents = 4
    else:
        intervention_str = ""

    if baseline:
        intervention_str = "_baseline"
        exp_dir = "exp1"

    for dataset in datasets:
        print( f"Processing {exp_dir}/out/output_{dataset}{intervention_str}{llm_judge_str}")
        with open(f"{exp_dir}/out/output_{dataset}{intervention_str}{llm_judge_str}_repeat1-stats.json", "r") as f:
            stats1 = json.load(f) 
        with open(f"{exp_dir}/out/output_{dataset}{intervention_str}{llm_judge_str}_repeat2-stats.json", "r") as f:
            stats2 = json.load(f) 
        with open(f"{exp_dir}/out/output_{dataset}{intervention_str}{llm_judge_str}_repeat3-stats.json", "r") as f:
            stats3 = json.load(f) 


        turning_points1, turning_points2, turning_points3 = 0, 0, 0
        avg_scores_per_turn = [0] * NUM_TURNS
        std_dev_per_turn = [0] * NUM_TURNS
        avg_scores_per_turn_delta = [0] * NUM_TURNS
        avg_scores_per_turn_delta_std_dev = [0] * NUM_TURNS
        avg_scores_per_turn_individual = [[0] * 3 for _ in range(NUM_TURNS)]
        avg_clockSeconds1, avg_clockSeconds2, avg_clockSeconds3 = 0, 0, 0
        range(2, 7*num_agents, num_agents)

        if not baseline:
            with open(f"{exp_dir}/out/output_{dataset}{intervention_str}{llm_judge_str}_repeat1-eval.json", "r") as f:
                eval_data1 = json.load(f)
                turning_points1 = calculate_avg_turning_points(eval_data1)
                avg_scores_per_turn1 = [0] * NUM_TURNS
                scores = [[[message["scores"][metrics[dataset][0]] for message in discussion["globalMemory"] if message["turn"] == turn and message.get("scores") is not None] for discussion in eval_data1] for turn in range(1, NUM_TURNS + 1)]
                for i, s in enumerate(scores):
                    avg_scores_per_turn1[i] = np.mean(s)
                avg_clockSeconds1 = np.mean([discussion["clockSeconds"] for discussion in eval_data1])
            with open(f"{exp_dir}/out/output_{dataset}{intervention_str}{llm_judge_str}_repeat2-eval.json", "r") as f:
                eval_data2 = json.load(f) 
                turning_points2 = calculate_avg_turning_points(eval_data2)
                avg_scores_per_turn2 = [0] * NUM_TURNS
                scores = [[[message["scores"][metrics[dataset][0]] for message in discussion["globalMemory"] if message["turn"] == turn and message.get("scores") is not None] for discussion in eval_data2] for turn in range(1, NUM_TURNS + 1)]
                for i, s in enumerate(scores):
                    avg_scores_per_turn2[i] = np.mean(s)
                avg_clockSeconds2 = np.mean([discussion["clockSeconds"] for discussion in eval_data2])
            with open(f"{exp_dir}/out/output_{dataset}{intervention_str}{llm_judge_str}_repeat3-eval.json", "r") as f:
                eval_data3 = json.load(f) 
                turning_points3 = calculate_avg_turning_points(eval_data3)
                avg_scores_per_turn3 = [0] * NUM_TURNS
                scores = [[[message["scores"][metrics[dataset][0]] for message in discussion["globalMemory"] if message["turn"] == turn and message.get("scores") is not None] for discussion in eval_data3] for turn in range(1, NUM_TURNS + 1)]
                for i, s in enumerate(scores):
                    avg_scores_per_turn3[i] = np.mean(s)
                avg_clockSeconds3 = np.mean([discussion["clockSeconds"] for discussion in eval_data3])

            eval_data[dataset] = eval_data1 + eval_data2 + eval_data3
            eval_data_seperated[dataset] = [eval_data1, eval_data2, eval_data3]
        
            for i in range(1, NUM_TURNS + 1):
                # Get scores for each run at this turn
                scores1 = [[message["scores"][metrics[dataset][0]] for message in discussion["globalMemory"] if message["turn"] == i and message.get("scores") is not None] for discussion in eval_data1]
                scores2 = [[message["scores"][metrics[dataset][0]] for message in discussion["globalMemory"] if message["turn"] == i and message.get("scores") is not None] for discussion in eval_data2]
                scores3 = [[message["scores"][metrics[dataset][0]] for message in discussion["globalMemory"] if message["turn"] == i and message.get("scores") is not None] for discussion in eval_data3]
                
                # Replace None with 0
                scores1 = [0 if score is None else score for score in scores1]
                scores2 = [0 if score is None else score for score in scores2]
                scores3 = [0 if score is None else score for score in scores3]
                
                # Calculate mean for each run
                mean1 = np.mean(scores1) if scores1 else 0
                mean2 = np.mean(scores2) if scores2 else 0
                mean3 = np.mean(scores3) if scores3 else 0
                
                # Calculate overall mean and std dev across runs
                avg_scores_per_turn[i-1] = np.mean([mean1, mean2, mean3])
                avg_scores_per_turn_individual[i-1] = [mean1, mean2, mean3]
                std_dev_per_turn[i-1] = np.std([mean1, mean2, mean3])
            
            avg_scores_per_turn_delta = np.mean([avg_scores_per_turn_individual[6][0] - avg_scores_per_turn_individual[0][0], avg_scores_per_turn_individual[6][1] - avg_scores_per_turn_individual[0][1], avg_scores_per_turn_individual[6][2] - avg_scores_per_turn_individual[0][2]])
            avg_scores_per_turn_delta_std_dev = np.std([avg_scores_per_turn_individual[6][0] - avg_scores_per_turn_individual[0][0], avg_scores_per_turn_individual[6][1] - avg_scores_per_turn_individual[0][1], avg_scores_per_turn_individual[6][2] - avg_scores_per_turn_individual[0][2]])

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
                "avg_scores_per_turn_delta": avg_scores_per_turn_delta,
                "avg_scores_per_turn_delta_std_dev": avg_scores_per_turn_delta_std_dev,
                "std_dev_scores_per_turn": std_dev_per_turn,
                "avg_clockSeconds": (avg_clockSeconds1 + avg_clockSeconds2 + avg_clockSeconds3) / 3
            }

        stats[dataset] = overall_stats

    output_path = f"exp3/figures/_eval_{dataset}_{intervention}{llm_judge_str}_stats.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=4)
    return stats, eval_data, eval_data_seperated

def main():

    print("-> Processing MALLM data")
    stats_baseline, _, eval_data_baseline_sep = get_experiment_stats_and_eval_data(llm_judge = False, intervention = None, dataset_to_process = "mmlu_pro", baseline = True)
    stats_normal, _, eval_data_normal_sep = get_experiment_stats_and_eval_data(llm_judge = False, intervention = None, dataset_to_process = "mmlu_pro")

    stats_policy, _, eval_data_policy_sep = get_experiment_stats_and_eval_data(llm_judge = False, intervention = "policy", dataset_to_process = "mmlu_pro")
    stats_regenerate, _, eval_data_regenerate_sep = get_experiment_stats_and_eval_data(llm_judge = False, intervention = "regenerate", dataset_to_process = "mmlu_pro")

    stats_policy_judge, eval_data_policy_judge, eval_data_policy_judge_sep = get_experiment_stats_and_eval_data(llm_judge = True, intervention = "policy", dataset_to_process = "mmlu_pro")
    stats_regenerate_judge, _, eval_data_regenerate_judge_sep = get_experiment_stats_and_eval_data(llm_judge = True, intervention = "regenerate", dataset_to_process = "mmlu_pro")

    print_overall_stats(stats_normal, stats_policy, stats_regenerate, stats_policy_judge, stats_regenerate_judge)
    successful_samples(eval_data_normal_sep, eval_data_policy_sep, eval_data_regenerate_sep, eval_data_policy_judge_sep, eval_data_regenerate_judge_sep)
    mallm_vs_baseline([stats_baseline, stats_normal, stats_policy_judge, stats_regenerate_judge])

if __name__ == "__main__":
    fire.Fire(main)