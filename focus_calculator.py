import os
import json
import nltk
nltk.download('punkt')

original_print = print

with open("metrics_for_figures.json", "r") as metrics_file:
    metrics = json.load(metrics_file)

class FocusCalculator:

    def __init__(self) -> None:
        pass

    @staticmethod
    def calculate_per_turn(sample) -> tuple[list[float], list[float], float, list[str]]:
        '''
        Calculates the focus score per turn and the total focus score for a given sample.
        Returns:
            tuple[list[float], list[float], float, list[str]]: A tuple containing the scores per turn, the theta (focus score) per turn, the total focus score, and the solutions per turn.
        '''
        votesEachTurn = sample["votesEachTurn"]
        scores_per_turn = []
        theta_per_turn = []
        solutions_per_turn = []
        for i, turn in enumerate(votesEachTurn):
            scores_per_turn.append(votesEachTurn[turn]['alterations']['public']['score'][metrics[os.path.splitext(os.path.basename(sample["dataset"]))[0]][0] + "-public"])
            solutions_per_turn.append(votesEachTurn[turn]['alterations']['public']['final_answer'])
            if scores_per_turn[-1] is None:
                scores_per_turn[-1] = 0
            if len(scores_per_turn) > 1:
                theta_per_turn.append(scores_per_turn[-1] - scores_per_turn[-2])
            else: 
                theta_per_turn.append(0)
        return scores_per_turn, theta_per_turn, sum(theta_per_turn), solutions_per_turn