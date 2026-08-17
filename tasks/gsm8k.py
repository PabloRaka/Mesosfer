"""
GSM8K evaluation.
https://huggingface.co/datasets/openai/gsm8k

Example problem instance:

Question:
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer:
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

Notice that GSM8K uses tool calls inside << >> tags.
"""

import re
from datasets import load_dataset
from tasks.common import Task


GSM_RE = re.compile(r"####\s*([\$]?\s*[\-0-9\.\,]+)")
def extract_answer(completion):
    """
    Extract numerical answer from completion supporting:
    - Official OpenAI GSM8K format (#### 123)
    - LaTeX boxed format (\\boxed{123})
    - Conversational answers ("The answer is 123", "is 123", "= 123")
    - Fallback to the last standalone number in the text
    """
    if not completion:
        return None

    # 1. Official OpenAI GSM8K marker: #### 123
    match = GSM_RE.search(completion)
    if match:
        raw = match.group(1).replace("$", "").replace(",", "").strip()
        if raw:
            return raw

    # 2. LaTeX boxed marker: \boxed{123}
    match = re.search(r"\\boxed\{\s*[\$]?\s*([\-0-9\.\,]+)\s*\}", completion)
    if match:
        raw = match.group(1).replace("$", "").replace(",", "").strip()
        if raw:
            return raw

    # 3. Conversational patterns: "The answer is 123" / "equals 123"
    matches = re.findall(r"(?:the answer is|equals?|=|\bis\b)\s*[\$]?\s*([\-0-9\.\,]+)", completion, re.IGNORECASE)
    if matches:
        raw = matches[-1].replace(",", "").strip().rstrip(".")
        if raw:
            return raw

    # 4. Fallback: find all numbers in text and pick the last one
    numbers = re.findall(r"[\-+]?\d+(?:\.\d+)?", completion)
    if numbers:
        return numbers[-1].rstrip(".")

    return None


class GSM8K(Task):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "GSM8K subset must be main|socratic"
        assert split in ["train", "test"], "GSM8K split must be train|test"
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ Get a single problem from the dataset. """
        row = self.ds[index]
        question = row['question'] # string of the question prompt
        answer = row['answer'] # string of the full solution and the answer after #### marker
        # Create and return the Conversation object
        # This is tricky because GSM8K uses tool calls, which we need to parse here.
        assistant_message_parts = []
        parts = re.split(r'(<<[^>]+>>)', answer)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                # This is a calculator tool call
                inner = part[2:-2]  # Remove << >>
                # Split on = to get expression and result
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                # Add the tool call as a part
                assistant_message_parts.append({"type": "calc", "text": expr})
                # Add the result as a part
                assistant_message_parts.append({"type": "calc_output", "text": result})
            else:
                # Regular text in between tool calls
                assistant_message_parts.append({"type": "text", "text": part})
        # Now put it all together
        messages = [
            {"role": "user", "content": question}, # note: simple string
            {"role": "assistant", "content": assistant_message_parts}, # note: list of parts (as dicts)
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        Note that:
        - the conversation has both user AND assistant message (containing the ground truth answer)
        - the assistant_response is usually the alternative assistant message achieved via sampling

        TODO: Technically, assistant_response should be a Message (either a string or a list of parts)
              We can handle this later possibly. For now just assume string.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        # First extract the ground truth answer
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        assert isinstance(assistant_message['content'], list), "This is expected to be a list of parts"
        last_text_part = assistant_message['content'][-1]['text'] # this contains the final answer in GSM8K
        # Extract both the ground truth answer and the predicted answer
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        if ref_num is None or pred_num is None:
            return 0
        if pred_num == ref_num:
            return 1
        try:
            import math
            return int(math.isclose(float(pred_num), float(ref_num), rel_tol=1e-4, abs_tol=1e-4))
        except (ValueError, OverflowError):
            return int(pred_num == ref_num)

    def reward(self, conversation, assistant_response):
        """
        Used during RL. To keep things simple, just re-use the evaluation above.
        Later this could be made more complex (e.g. format matching etc.)
        """
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float
