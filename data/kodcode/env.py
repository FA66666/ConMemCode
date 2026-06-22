from common.utils.code_utils import PyExecutor, collect_python_program, rename_function
from data.base_env import StaticEnv

KODCODE_INSTRUCTIONS = "Write a Python function according to the given task"


def _resolve_function_name(test_info) -> str:
    if not test_info or not isinstance(test_info, list):
        return ""
    first = test_info[0]
    if not isinstance(first, dict):
        return ""
    return str(first.get("function_name", "")).strip()

class KodCodeEnv(StaticEnv):

    def __init__(self, config):
        super().__init__(config)
        self.feedback_detail = None

    def set_env(self, task_config: dict) -> tuple[str, str]:

        self.prompt = task_config.get("prompt")
        self.test = task_config.get("test")
        self.test_info = task_config.get("test_info")

        return KODCODE_INSTRUCTIONS, self.prompt

    def step(self, action: str) -> tuple[str, float, bool]:

        py_executor = PyExecutor()
        collected_answer = collect_python_program((action or "").strip())
        function_name = _resolve_function_name(self.test_info)
        renamed_answer = rename_function(collected_answer, function_name) if function_name else collected_answer

        is_passing, feedback, results = py_executor.execute(renamed_answer, [self.test])
        score = sum(results) / len(results) if results else 0.0

        if score >= 1.0:
            summary = "All tests passed."
        elif score > 0:
            summary = f"Partial pass: {score:.1%}"
        else:
            summary = "All tests failed."

        self.feedback_detail = {
            "summary": summary,
            "score": score,
            "full_feedback": feedback,
            "test_passed": is_passing,
            "test_results": results[0] if results else False,
        }

        self.reward = score
        if self.reward == 1.0:
            self.observation = "True!"
            self.done = True
        else:
            self.observation = "False!"
            self.done = False

        return self.observation, self.reward, self.done

    @classmethod
    def compute_reward(cls, completions: list[str], test: list[str], test_info: list, **kwargs) -> list[float]:

        py_executor = PyExecutor()
        scores = []
        for completion, t, tf in zip(completions, test, test_info):
            collected_answer = collect_python_program((completion or "").strip())
            function_name = _resolve_function_name(tf)
            renamed_answer = rename_function(collected_answer, function_name) if function_name else collected_answer
            _, _, results = py_executor.execute(renamed_answer, [t])

            score = sum(results) / len(results) if results else 0.0
            scores.append(score)

        return scores
