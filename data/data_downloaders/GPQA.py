import json
import random
import uuid
from typing import Optional

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample


class GPQADownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(
        self, sample_size: Optional[int] = None, hf_token: Optional[str] = None, trust_remote_code: bool = False
    ):
        super().__init__(
            name="gpqa",
            dataset_name="Idavidrein/gpqa",
            version="gpqa_extended",
            sample_size=sample_size,
            hf_token=hf_token,
            trust_remote_code=trust_remote_code
        )

    def process_data(self) -> list[InputExample]:
        data = self.shuffle_and_select()
        input_examples = []

        for sample in data.iter(batch_size=1):
            answers, correct_answer = self._format_answers(sample)
            correct_answer_index = answers.index(json.dumps(correct_answer))
            correct_answer_label = f"{chr(65 + correct_answer_index)}) {correct_answer}"

            question_text = self._clean_text(sample["Question"][0])
            formatted_answers = self._format_answer_choices(answers)
            question_and_answers = f"{question_text}\n\n" + "\n".join(formatted_answers)

            input_examples.append(
                InputExample(
                    example_id=str(uuid.uuid4()),
                    dataset_id=sample["Canary String"][0],
                    inputs=[question_and_answers],
                    context=None,
                    references=[correct_answer_label],
                )
            )
        return input_examples

    @staticmethod
    def _format_answers(sample: InputExample):
        answers = [json.dumps(sample[f"Incorrect Answer {i}"][0]) for i in range(1, 4)]
        correct_answer = sample["Correct Answer"][0]
        answers.insert(0, json.dumps(correct_answer))
        random.shuffle(answers)
        return answers, correct_answer
