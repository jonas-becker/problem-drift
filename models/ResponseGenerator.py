import logging
from models.Chat import Chat
from openai import OpenAI
logger = logging.getLogger("mallm")


class ResponseGenerator():

    def __init__(self, endpoint_url: str):
        self.llm = Chat(
            client=OpenAI(
                base_url=endpoint_url, api_key="-"
            ),
            model="tgi",
        )

    def extract_focus_and_reason(self, response: str) -> tuple[float, str]:
        focus = 0
        for word in response.split():
            word = ''.join(filter(str.isdigit, word)) if any(char.isdigit() for char in word) else word
            if word.isdigit() or (word.startswith('-') and word[1:].isdigit()):
                focus = int(word)
                if len(response.split(word, 1)) > 1:
                    response = response.split(word, 1)[1].strip()
                break
        
        if not 0 <= focus <= 10: 
            return None, "Focus value is out of range"
        
        response = response.replace("[Reason]", "").replace("[Focus]", "")
        response = response.replace("[reason]", "").replace("[focus]", "")
        return focus, response
    
    def generate_reason(
        self,
        task_instruction: str,
        input_str: str,
        messages: dict[str, str]
    ):
        prompt = [
            {
                "role": "system",
                "content": "You are given six recent messages of a discussion that aims to solve a task. Your task is to determine why the discussion is going badly. You should output a number between 0 and 10, where 0 means the discussion is going very badly and 10 means the discussion is going very well. You should also output a reason for your rating. Name the explicit moment in the discussion which is crucial to your rating. The agreement between the participants is not a valid reason. \n\nTask Instruction: " + task_instruction + "\n\nInput: " + input_str + "\n\nThis is the conversation so far: ",
            }
        ]

        for p in messages.keys():
            prompt.append(
                {
                    "role": "user",
                    "content": f"{p}: {messages[p]}",
                }
            )

        prompt.append(
            {
                "role": "user",
                "content": "\nYour output should be in the following format: [Focus] <number between -10 and 10> [Reason] <reason for your rating> \nMake sure explain your rating briefly and in a way that is easy to understand. Do not write anything else.",
            }
        )

        res = self.llm.invoke(prompt)

        focus, reason = self.extract_focus_and_reason(res)
        return focus, reason
