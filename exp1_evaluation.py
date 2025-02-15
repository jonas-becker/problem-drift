from mallm.evaluation.evaluator import Evaluator
import glob
import os
import json
import fire
import logging
import nltk
nltk.download('punkt')

logger = logging.getLogger("mallm")

original_print = print

def main():
    input_files = glob.glob(os.path.join("exp1/out/", "*.json"))

    with open("metrics.json", "r") as metrics_file:
        metrics = json.load(metrics_file)

    for input_file in input_files:
        if "-eval" in input_file or "-stats" in input_file or input_file+"-eval" in input_files:
            continue
        logger.info("Evaluating " + input_file)
        
        m = []
        for metric in metrics.keys():
            if metric in input_file:
                m = metrics[metric]
        if "baseline" in input_file:
            evaluator = Evaluator(input_file_path=input_file, metrics=m, extensive=False)
        else:
            evaluator = Evaluator(input_file_path=input_file, metrics=m, extensive=True)
        evaluator.process()

if __name__ == "__main__":
    fire.Fire(main)