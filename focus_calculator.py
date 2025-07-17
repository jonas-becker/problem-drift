import os
import json
import nltk
from sentence_transformers import SentenceTransformer
nltk.download('punkt')

original_print = print

with open("metrics_for_figures.json", "r") as metrics_file:
    metrics = json.load(metrics_file)

class FocusCalculator:

    def __init__(self, model_name='all-MiniLM-L6-v2') -> None:
        """
        Initialize FocusCalculator with specified model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use.
                             Options include:
                             - 'all-MiniLM-L6-v2' (default, fast)
                             - 'microsoft/DialoGPT-medium' (ModernBERT-like)
                             - 'sentence-transformers/all-mpnet-base-v2' (high quality)
                             - 'intfloat/e5-base-v2' (state-of-the-art)
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def calculate_per_turn(self, sample) -> tuple[list[float], list[float], float, list[str]]:
        '''
        Calculates the focus score per turn and the total focus score for a given sample.
        Returns:
            tuple[list[float], list[float], float, list[str]]: A tuple containing the scores per turn, the theta (focus score) per turn, the total focus score, and the solutions per turn.
        '''
        votesEachTurn = sample["votesEachTurn"]
        scores_per_turn = []
        theta_per_turn = []
        solutions_per_turn = []
        if votesEachTurn is not None:
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
    
    def calculate_per_turn_embedding_similarity(self, sample) -> tuple[list[float], list[float], float, list[str]]:
        '''
        Calculates the embedding similarity per turn and the total embedding similarity for a given sample.
        Returns:
            tuple[list[float], list[float], float, list[str]]: A tuple containing the similarity scores per turn, the theta (similarity change) per turn, the total similarity score, and the solutions per turn.
        '''
        globalMemory = sample["globalMemory"]
        similarity_scores_per_turn = []
        theta_per_turn = []
        solutions_per_turn = []
        first_turn_embedding = None

        # Get solutions per turn for reference
        votesEachTurn = sample.get("votesEachTurn", {})
        if votesEachTurn:
            for turn in votesEachTurn:
                solutions_per_turn.append(votesEachTurn[turn]['alterations']['public']['final_answer'])

        for turn in range(1, 7):
            messages_of_turn = [m["message"] for m in globalMemory if m["turn"] == turn]
            
            if not messages_of_turn:
                similarity_scores_per_turn.append(0)
                theta_per_turn.append(0)
                continue
                
            # Calculate embeddings for messages in this turn
            embeddings = []
            for message in messages_of_turn:
                embedding = self.model.encode(message)
                embeddings.append(embedding)
            
            # Calculate average embedding for this turn
            if embeddings:
                avg_embedding = sum(embeddings) / len(embeddings)
                if turn == 1:
                    first_turn_embedding = avg_embedding
                
                # Calculate similarity with previous turn (if exists)
                if turn > 1 and len(similarity_scores_per_turn) > 0:
                    # Use cosine similarity between current and previous average embeddings
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarity_scores_per_turn[-1]  # Store the previous avg embedding
                    similarity = cosine_similarity([avg_embedding], [first_turn_embedding])[0][0]
                    similarity_scores_per_turn.append(similarity)
                    
                    # Calculate theta (change in similarity)
                    if len(similarity_scores_per_turn) > 1:
                        theta_per_turn.append(similarity_scores_per_turn[-1] - similarity_scores_per_turn[-2])
                    else:
                        theta_per_turn.append(0)
                else:
                    # First turn - no previous to compare with
                    similarity_scores_per_turn.append(1.0)  # Perfect similarity with itself
                    theta_per_turn.append(0)
            else:
                similarity_scores_per_turn.append(0)
                theta_per_turn.append(0)
        
        return similarity_scores_per_turn, theta_per_turn, sum(theta_per_turn), solutions_per_turn

            