import glob
import os
import json
import fire
import pandas as pd
import nltk
import requests
import sys
import httpx
from focus_calculator import FocusCalculator
from pandas import DataFrame
from models.ResponseGenerator import ResponseGenerator
nltk.download('punkt')

original_print = print

ENDPOINT_URL = "http://134.76.18.30:8080/v1"
model = ResponseGenerator(ENDPOINT_URL)

with open("metrics_for_figures.json", "r") as metrics_file:
    metrics = json.load(metrics_file)

def generate_focus_and_reason(sample, highest_theta_turn, number_of_relevant_turns, highest_theta):
    lower_bound = highest_theta_turn - number_of_relevant_turns if highest_theta_turn - number_of_relevant_turns >= 1 else 1
    upper_bound = highest_theta_turn
    messages = dict()
    for mem in sample["globalMemory"]:
        if lower_bound <= mem["turn"] <= upper_bound:
            messages[mem['persona']] = mem['message']

    try:
        focus, reason = model.generate_reason(sample["instruction"], sample["input"][0], messages)
    except Exception:
        return None, None
    return focus, reason

def pretty_print_discussion(sample):
    pretty_discussion = ""
    pretty_discussion += f"--------------------------------------------\n Task Instruction:\n{str(sample['instruction'])} \n\nInput:\n{str(sample['input'][0])} \n\nContext:\n{str(sample['context'])} \n\n<p style=\"color: blue;\">--> FINAL SOLUTION:\n{str(sample['finalAnswer'])} \n\n--> REFERENCE:\n{sample['references'][0] if len(sample['references']) > 0 else "None"}</p>\n\nTHE DISCUSSION:\n"
    for mem in sample["globalMemory"]:
        pretty_discussion += f"--------------------------------------------(Turn {mem["turn"]})\n <p style=\"color: red;\">{mem['persona'].upper()}:</p> {mem['message']} \n\n<p style=\"color: blue;\">EXTRACTED SOLUTION:\n{mem['solution']}</p>\n"
    return pretty_discussion + "\n\n<p style=\"color: green;\">=====================================================\n=====================================================\nNEXT DISCUSSION\n=====================================================</p>\n\n"

def pretty_print_discussion_only(globalMemory):
    pretty_discussion = ""
    for mem in globalMemory:
        pretty_discussion += f"--------------------------------------------(Turn {mem["turn"]})\n{mem['persona'].upper()}: {mem['message']} \n\nEXTRACTED SOLUTION:\n{mem['solution']}\n"
    return pretty_discussion 

def pretty_print_solutions_only(globalMemory):
    pretty_discussion = ""
    for mem in globalMemory:
        pretty_discussion += f"--------------------------------------------(Turn {mem["turn"]})\n{mem['persona'].upper()}: {mem['solution']}\n"
    return pretty_discussion

def pretty_print_scores_per_turn(sample):
    votesEachTurn = sample["votesEachTurn"]
    pretty_discussion = ""
    for i, turn in enumerate(votesEachTurn):
        pretty_discussion += f"TURN {i+1}: {votesEachTurn[turn]['alterations']['public']['final_answer']}\n {metrics[os.path.splitext(os.path.basename(sample["dataset"]))[0]][0]}: {votesEachTurn[turn]['alterations']['public']['score'][metrics[os.path.splitext(os.path.basename(sample["dataset"]))[0]][0] + "-public"]}\n"
    sample["scoresEachTurn"] = pretty_discussion
    return sample 

def pretty_print_scores_per_turn_string(sample):
    votesEachTurn = sample["votesEachTurn"]
    pretty_discussion = f"{metrics[os.path.splitext(os.path.basename(sample["dataset"]))[0]][0]}\n\n"
    for i, turn in enumerate(votesEachTurn):
        pretty_discussion += f"TURN {i+1}: {votesEachTurn[turn]['alterations']['public']['score'][metrics[os.path.splitext(os.path.basename(sample["dataset"]))[0]][0] + "-public"]}\n\n"
    return pretty_discussion

def extract_partly(sample, highest_theta_turn, number_of_relevant_turns, highest_theta):
    lower_bound = highest_theta_turn - number_of_relevant_turns if highest_theta_turn - number_of_relevant_turns >= 1 else 1
    upper_bound = highest_theta_turn
    pretty_discussion = ""
    pretty_discussion += f"--------------------------------------------\n Task Instruction:\n{str(sample['instruction'])} \n\nInput:\n{str(sample['input'][0])} \n\nContext:\n{str(sample['context'])} \n\n<p style=\"color: blue;\">--> FINAL SOLUTION:\n{str(sample['finalAnswer'])} \n\n--> REFERENCE:\n{sample['references'][0] if len(sample['references']) > 0 else "None"}</p>\n\nDrift Strength of Turn {highest_theta_turn + 1}: {highest_theta}\n\nTHE DISCUSSION:\n"
    for mem in sample["globalMemory"]:
        if lower_bound <= mem["turn"] <= upper_bound:
            pretty_discussion += f"--------------------------------------------(Turn {mem["turn"]})\n <p style=\"color: red;\">{mem['persona'].upper()}:</p> {mem['message']} \n\n<p style=\"color: blue;\">EXTRACTED SOLUTION:\n{mem['solution']}</p>\n"
    return pretty_discussion + "\n\n<p style=\"color: green;\">=====================================================\n=====================================================\nNEXT DISCUSSION\n=====================================================</p>\n\n"

def extract_partly_solutions(sample, highest_theta_turn, number_of_relevant_turns, highest_theta):
    lower_bound = highest_theta_turn - number_of_relevant_turns if highest_theta_turn - number_of_relevant_turns >= 1 else 1
    upper_bound = highest_theta_turn
    pretty_discussion = ""
    pretty_discussion += f"--------------------------------------------\n Task Instruction:\n{str(sample['instruction'])} \n\nInput:\n{str(sample['input'][0])} \n\nContext:\n{str(sample['context'])} \n\n<p style=\"color: blue;\">--> FINAL SOLUTION:\n{str(sample['finalAnswer'])} \n\n--> REFERENCE:\n{sample['references'][0] if len(sample['references']) > 0 else "None"}</p>\n\nDrift Strength of Turn {highest_theta_turn + 1}: {highest_theta}\n\nTHE DISCUSSION:\n"
    for mem in sample["globalMemory"]:
        if lower_bound <= mem["turn"] <= upper_bound:
            pretty_discussion += f"--------------------------------------------(Turn {mem["turn"]})\n <p style=\"color: red;\">{mem['persona'].upper()}:</p> {mem['solution']}</p>\n"
    return pretty_discussion + "\n\n<p style=\"color: green;\">=====================================================\n=====================================================\nNEXT DISCUSSION\n=====================================================</p>\n\n"

def extract_to_dataframe_raw(extracted_samples_all, input_file):
    df = DataFrame(extracted_samples_all)
    df['discussionString'] = df['globalMemory'].apply(lambda x: pretty_print_discussion_only(x))
    df['solutionsString'] = df['globalMemory'].apply(lambda x: pretty_print_solutions_only(x))
    df['references'] = df['references'].apply(lambda x: x[0] if len(x)>0 else None)
    df['input'] = df['input'].apply(lambda x: x[0])
    df['dataset'] = df['dataset'].apply(lambda x: x.split("/")[-1])
    df['personas'] = df['personas'].apply(lambda x: x[0]["persona"] + ", " + x[1]["persona"] + ", " + x[2]["persona"])
    df['context'] = df['context'].apply(lambda x: ', '.join(x) if x else None)
    df = df.apply(pretty_print_scores_per_turn, axis=1)
    df = df.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
    df.to_csv(input_file.replace("-eval.json" ,"-extracted.csv"), index=False)
    df.to_excel(input_file.replace("-eval.json" ,"-extracted.xlsx"), index=False)

def extract_to_dataframe(extracted_samples, number_of_relevant_turns, input_file):
    message_tuples = [('Extracted Messages', '1'), ('Extracted Messages', '2'), ('Extracted Messages', '3'), ('Extracted Messages', '4'), ('Extracted Messages', '5'), ('Extracted Messages', '6')]
    solutions_tuples = [('Voted Solution (Before)', ''), ('Voted Solution (After)', '')]
    other_columns = ['Sample ID', 'Dataset', 'Instruction', 'Input', 'Context', 'Reference', 'Personas', 'Persona Diversity', 'Strongest Drift Turn', 'Drift Strength', 'Drift Strength (Per Turn)', 'Scores', 'Focus (by LLM)', 'Reason (by LLM)', 'Complete Discussion']

    columns = pd.MultiIndex.from_tuples([(col, '') for col in other_columns] + message_tuples + solutions_tuples)

    df = pd.DataFrame(columns=columns)

    for sample in extracted_samples:
        highest_theta_turn = sample["highestThetaTurn"]

        lower_bound = highest_theta_turn - number_of_relevant_turns if highest_theta_turn - number_of_relevant_turns >= 1 else 1
        upper_bound = highest_theta_turn

        row = {}
        row[('Dataset', '')] = sample['dataset']
        row[('Sample ID', '')] = sample['exampleId']
        row[('Instruction', '')] = sample['instruction']
        row[('Input', '')] = sample['input'][0]
        row[('Context', '')] = None if not sample['context'] else sample['context'][0]
        row[('Reference', '')] = None if not sample['references'] else sample['references'][0]
        row[('Personas', '')] = ", ".join([persona['persona'] for persona in sample['personas']])
        row[('Persona Diversity', '')] = sample['persona_diversity']
        row[('Strongest Drift Turn', '')] = highest_theta_turn
        row[('Drift Strength', '')] = sample["driftStrength"]
        row[('Drift Strength (Per Turn)', '')] = sample["thetasPerTurn"]
        row[('Scores', '')] = pretty_print_scores_per_turn_string(sample)
        row[('Complete Discussion', '')] = pretty_print_discussion_only(sample['globalMemory'])
        row[('All Solutions During Discussion', '')] = pretty_print_solutions_only(sample['globalMemory'])
        row[('Persona Diversity', '')] = sample['persona_diversity']
        row[('Focus (by LLM)', '')] =  sample['focus']
        row[('Reason (by LLM)', '')] = str(sample['reason'])
        row[('Voted Solution (Before)', '')] = str(sample["solutionsPerTurn"][highest_theta_turn-2])
        row[('Voted Solution (After)', '')] = str(sample["solutionsPerTurn"][highest_theta_turn-1])

        i = 0
        for mem in sample['globalMemory']:
            if lower_bound <= mem["turn"] <= upper_bound:
                i += 1
                row[('Extracted Messages', str(i))] = mem['message']

        df = pd.concat([df, pd.DataFrame([row], columns=df.columns)], ignore_index=True)
    
    df = df.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
    df.to_csv(input_file.replace("-eval.json" ,"-extractedFormatted.csv"))
    df.to_excel(input_file.replace("-eval.json" ,"-extractedFormatted.xlsx"))
    print(f"Extracted {input_file} to datframe")
    return df

def make_csv_from_extracted_discussions(input_files):
    extracted_samples_all = []
    for input_file in input_files:
        print("--> Extracting " + input_file)
        with open(os.path.join(input_file), "r") as f:
            eval_data = json.load(f)
        extracted_samples = []
        extracted_pretty_text, extracted_pretty_solutions_text = "", ""
        dataset = os.path.splitext(os.path.basename(eval_data[0]["dataset"]))[0]
        print("Dataset: " + str(dataset))

        scores_per_turn, thetas_per_turn, total_thetas, solutions_per_turn = [], [], [], []

        for sample in eval_data:
            scores, thetas, total_theta, solutions = FocusCalculator.calculate_per_turn(sample)
            thetas_per_turn.append(thetas)
            scores_per_turn.append(scores)
            solutions_per_turn.append(solutions)
            total_thetas.append(total_theta)

        highest_total_thetas_indices = sorted(range(len(total_thetas)), key=lambda i: total_thetas[i], reverse=True)[-20:]

        for i in highest_total_thetas_indices:
            sample = eval_data[i]
            highest_theta_turn = thetas_per_turn[i].index(min(thetas_per_turn[i]))+1
            sample["solutionsPerTurn"] = solutions_per_turn[i]
            sample["thetasPerTurn"] = thetas_per_turn[i]
            sample["highestThetaTurn"] = highest_theta_turn
            sample["driftStrength"] = min(thetas_per_turn[i])
            sample["focus"], sample["reason"] = generate_focus_and_reason(sample, highest_theta_turn, 1, min(thetas_per_turn[i]))
            extracted_samples.append(sample)
            extracted_samples_all.append(sample)

            extracted_pretty_text += f"=============== SAMPLE {sample["exampleId"]}, {sample["dataset"]} \n\n" + extract_partly(sample, highest_theta_turn, 1, min(thetas_per_turn[i]))
            extracted_pretty_solutions_text += f"=============== SAMPLE {sample["exampleId"]}, {sample["dataset"]} \n\n" + extract_partly_solutions(sample, highest_theta_turn, 1, min(thetas_per_turn[i]))

        extract_to_dataframe(extracted_samples, 1, input_file)


        with open(input_file.replace("-eval" ,"-extracted"), "w") as output_file:
            json.dump(extracted_samples, output_file)
            print(f"{len(extracted_samples)} extracted samples saved.")
        
        extracted_html_text = extracted_pretty_text.replace("\n", "<br/>")
        extracted_html_text_solutions = extracted_pretty_solutions_text.replace("\n", "<br/>")

        with open(input_file.replace("-eval.json" ,"-extracted.html"), "w", encoding='utf-8') as html_file:
            html_file.write("<html><body>" + extracted_html_text + "</body></html>")
            print("Extracted pretty text transformed to HTML and saved to an HTML file.")
        with open(input_file.replace("-eval.json" ,"-extractedsolutions.html"), "w", encoding='utf-8') as html_file:
            html_file.write("<html><body>" + extracted_html_text_solutions + "</body></html>")
            print("Extracted pretty text (solutions) transformed to HTML and saved to an HTML file.")

        extract_to_dataframe_raw(extracted_samples_all, input_file)
    
    with open("exp1/out/all-extracted.json", "w") as output_file:
        json.dump(extracted_samples_all, output_file)
        print("All extracted samples saved to a single file.")

    with open("exp1/out/llm_reasons.txt", "w") as txt_file:
        for sample in extracted_samples_all:
            reason = sample.get("reason", "")
            if not reason:
                reason = "None"
            txt_file.write(f"{reason.encode('ascii', 'ignore').decode('ascii')}\n\n")
        print("Focus and reason columns extracted and saved to a formatted txt file.")

def main():
    
    try:
        print("Testing availability of the endpoint...")
        page = requests.head(ENDPOINT_URL.replace("/v1", ""))
        print("Status: " + str(page.status_code))
        assert page.status_code == 200
    except Exception as e:
        print("HTTP Error: Could not connect to the provided endpoint url.")
        print(e)
        sys.exit(1)

    with httpx.Client():
        input_files = glob.glob(os.path.join("exp1/out/", "*-eval.json"))
        input_files = [file for file in input_files if "baseline" not in file]
        make_csv_from_extracted_discussions(input_files)
    
if __name__ == "__main__":
    fire.Fire(main)