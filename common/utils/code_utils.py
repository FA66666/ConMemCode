#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ast
import os
from typing import List, Tuple, Any, Optional
import re
import multiprocessing
from multiprocessing.connection import Connection

ExecuteResult = Tuple[bool, str, Tuple[bool]]


class TestExecutionError(Exception):
    """Custom test execution error with detailed diagnostics."""
    
    def __init__(self, error_info: dict):
        self.error_type = error_info.get("type", "Unknown")
        self.error_message = error_info.get("message", "")
        self.traceback = error_info.get("traceback", "")
        super().__init__(f"{self.error_type}: {self.error_message}")
    
    def format_error(self) -> str:
        """Format error details for display."""
        lines = [
            f"Error Type: {self.error_type}",
            f"Message: {self.error_message}",
        ]
        if self.traceback:
            lines.append("Traceback:")
            lines.append(self.traceback)
        return "\n".join(lines)


def _strip_code_wrappers(block: str) -> str:
    """Remove common model-output wrappers while preserving the Python source."""
    text = (block or "").strip()
    if not text:
        return ""

    # Models sometimes put a markdown code block inside XML <code> tags, or
    # forget the closing fence. Strip repeated outer wrappers before parsing.
    while True:
        stripped = text.strip()
        fence_match = re.fullmatch(
            r"```(?:python|py)?[ \t\r\n]*(.*?)(?:```)?[ \t\r\n]*",
            stripped,
            re.DOTALL | re.IGNORECASE,
        )
        if fence_match:
            text = fence_match.group(1).strip()
            continue

        xml_match = re.fullmatch(
            r"(?:<implementation>\s*)?<code>\s*(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?\s*</code>\s*(?:</implementation>)?",
            stripped,
            re.DOTALL | re.IGNORECASE,
        )
        if xml_match:
            text = xml_match.group(1).strip()
            continue
        break

    # Remove stray fence-only lines that can remain after partial extraction.
    text = re.sub(r"(?im)^\s*```(?:python|py)?\s*$", "", text)
    text = re.sub(r"(?im)^\s*```\s*$", "", text)
    return text.strip()


def is_compilable_python(source: str) -> bool:
    try:
        ast.parse(source or "")
        return True
    except (SyntaxError, ValueError, TypeError, MemoryError):
        return False


def _extract_python_blocks(text_string: str) -> List[str]:
    xml_code_blocks = re.findall(
        r"<code>\s*<!\[CDATA\[(.*?)\]\]>\s*</code>",
        text_string,
        re.DOTALL,
    )
    if not xml_code_blocks:
        xml_code_blocks = re.findall(r"<code>(.*?)</code>", text_string, re.DOTALL)
    if xml_code_blocks:
        blocks = xml_code_blocks
    else:
        blocks = re.findall(
            r"```(?:python|py)?\s*(.*?)(?:```|\Z)",
            text_string,
            re.DOTALL | re.IGNORECASE,
        )
        if not blocks:
            blocks = [text_string]

    cleaned_blocks = [_strip_code_wrappers(block) for block in blocks if block and block.strip()]
    cleaned_blocks = [block for block in cleaned_blocks if block]
    if cleaned_blocks:
        return cleaned_blocks
    stripped = text_string.strip()
    return [_strip_code_wrappers(stripped)] if stripped else []


def extract_python_programs(text_string: str) -> List[str]:
    """Return full extracted Python program blocks without dropping structure."""
    return _extract_python_blocks(text_string)


def collect_python_program(text_string: str) -> str:
    """Collect extracted program blocks into one executable source string."""
    return "\n\n".join(extract_python_programs(text_string))


def contains_python_function_source(text_string: str) -> bool:
    programs = extract_python_programs(text_string)
    return any(re.search(r"^\s*def\s+\w+\s*\(", program, re.MULTILINE) for program in programs)


def extract_python_code(text_string: str) -> List[str]:
    # Historical helper used by some reporting utilities. It intentionally
    # extracts imports and function-level chunks rather than preserving the
    # whole program structure.
    code_blocks = _extract_python_blocks(text_string)
    results = []
    for block in code_blocks:
        imports = re.findall(r"^(?:from\s+\S+\s+import\s+\S+|import\s+\S+.*)$", block, re.MULTILINE)

        funcs = re.findall(r"(def\s+\w+\(.*?:[\s\S]*?)(?=^def\s|\Z)", block.strip(), re.MULTILINE)

        if imports:
            import_block = "\n".join(imports)
            if funcs:
                funcs = [import_block] + funcs
            else:
                funcs = [import_block]

        results.extend(funcs)

    return results

def rename_function(function: str, function_name: str) -> str:
    """
    Replace the name of the first function in `answer` with `function_name`.
    Only modifies the function name, keeps everything else intact.
    """
    pattern = r"def\s+(\w+)\s*\("

    new_answer = re.sub(pattern, f"def {function_name}(", function, count=1)
    return new_answer


def _exec_code_and_capture(code: str, conn: Connection, work_dir: Optional[str] = None):
    try:
        if work_dir is not None:
            os.makedirs(work_dir, exist_ok=True)
            os.chdir(work_dir)

        # Inject common libraries needed by the KodCode execution environment.
        import builtins
        import io
        import sys
        from dataclasses import dataclass
        
        # Create a mock capsys fixture.
        @dataclass
        class CaptureResult:
            out: str
            err: str
        
        class MockCapsys:
            def __init__(self):
                self._out = io.StringIO()
                self._err = io.StringIO()
                self._original_stdout = sys.stdout
                self._original_stderr = sys.stderr
            
            def readouterr(self):
                return CaptureResult(out=self._out.getvalue(), err=self._err.getvalue())
            
            def __enter__(self):
                sys.stdout = self._out
                sys.stderr = self._err
                return self
            
            def __exit__(self, *args):
                sys.stdout = self._original_stdout
                sys.stderr = self._original_stderr
        
        local_ns = {
            "__builtins__": builtins,
            "MockCapsys": MockCapsys,
        }
        
        # Pre-inject key standard-library modules so exec can use them.
        # These modules are frequently used in KodCode tasks.
        _stdlib_modules = [
            "re", "math", "random", "json", "itertools", "collections",
            "datetime", "statistics", "functools", "hashlib", "string",
            "inspect", "io", "sys", "os", "typing", "traceback",
            "collections.abc", "decimal", "fractions", "heapq", "bisect",
            "copy", "copyreg", "pickle", "pprint", "textwrap", "enum",
            "numbers", "pathlib", "dataclasses", "uuid", "time", "builtins",
        ]
        
        for mod_name in _stdlib_modules:
            try:
                mod = __import__(mod_name)
                local_ns[mod_name] = mod
            except ImportError:
                pass
        
        # Inject common typing aliases.
        try:
            import typing
            local_ns["typing"] = typing
            local_ns["List"] = typing.List
            local_ns["Tuple"] = typing.Tuple
            local_ns["Dict"] = typing.Dict
            local_ns["Set"] = typing.Set
            local_ns["Optional"] = typing.Optional
            local_ns["Union"] = typing.Union
            local_ns["Any"] = typing.Any
            local_ns["Callable"] = typing.Callable
            local_ns["Iterator"] = typing.Iterator
            local_ns["Iterable"] = typing.Iterable
        except ImportError:
            pass
        
        exec(code, local_ns)

        for name, func in local_ns.items():
            if callable(func) and name.startswith("test_"):
                # Check whether the function needs a capsys argument.
                import inspect
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                
                if "capsys" in params:
                    # Provide the mock capsys fixture.
                    capsys = MockCapsys()
                    with capsys:
                        func(capsys=capsys)
                else:
                    func()
        conn.send(True)
    except Exception as e:
        import traceback
        error_info = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        conn.send(error_info)
    finally:
        conn.close()


class PyExecutor:

    def _run_with_timeout(self, code: str, timeout: int, work_dir: Optional[str] = "./code_stuff") -> Any:
        parent_conn, child_conn = multiprocessing.Pipe()
        p = multiprocessing.Process(
            target=_exec_code_and_capture,
            args=(code, child_conn, work_dir)
        )
        
        p.start()
        p.join(timeout)

        if p.is_alive():
            p.kill()
            p.join()
            raise TimeoutError("Test execution timed out")

        if parent_conn.poll():
            result = parent_conn.recv()
            if isinstance(result, dict) and "type" in result:
                # New error format.
                raise TestExecutionError(result)
            elif isinstance(result, Exception):
                # Backward-compatible legacy format.
                raise result
            return result
        else:
            raise RuntimeError("Child process terminated unexpectedly without sending a result.")

    def execute(self, func: str, tests: List[str], timeout: int = 5, verbose: bool = True) -> ExecuteResult:
        success_tests = []
        failed_tests = []
        is_passing = True

        for test_code in tests:
            cleaned_test = re.sub(r"^\s*from\s+solution\s+import\s+\w+\s*", "", test_code, flags=re.MULTILINE)
            code_to_run = func + "\n" + cleaned_test
            try:
                self._run_with_timeout(code_to_run, timeout)
                success_tests.append(test_code)
            except TestExecutionError as e:
                # Detailed error output.
                error_detail = f"""
--- Test Failed ---
Test Code:
{test_code}

Error Type: {e.error_type}
Error Message: {e.error_message}
{chr(10) + 'Traceback:' + chr(10) + e.traceback if e.traceback else ''}
---
"""
                failed_tests.append(error_detail)
                is_passing = False
            except Exception as e:
                # Other exceptions, such as TimeoutError.
                failed_tests.append(f"""
--- Test Failed ---
Test Code:
{test_code}

Error Type: {type(e).__name__}
Error Message: {str(e)}
---
""")
                is_passing = False

        state = tuple(test in success_tests for test in tests)
        feedback = (
            "Tests passed:\n" + "\n".join(success_tests)
            + "\n\nTests failed:\n" + "\n".join(failed_tests)
        )
        return is_passing, feedback, state

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        cleaned_test = re.sub(r"^\s*from\s+solution\s+import\s+\w+\s*", "", test, flags=re.MULTILINE)
        code_to_run = func + "\n" + cleaned_test
        try:
            self._run_with_timeout(code_to_run, timeout)
            return True
        except Exception:
            return False
        
    def check_code_report(self, completions: list[str], tests: list[str], timeout: int = 5) -> tuple[list[str], list[float]]:
        def extract_failed_tests(text: str) -> str:
            match = re.search(r"Tests failed:\s*(.*)", text, re.DOTALL)
            return match.group(1).strip() if match else ""
        
        def extract_correct_function_name(text: str) -> str:
            match = re.search(r"from\s+solution\s+import\s+([a-zA-Z_]\w*)", text)
            return match.group(1) if match else ""

        reports = []
        avg_scores = [] 

        for completion, test_code_str in zip(completions, tests):
            collected_answer = collect_python_program(completion.strip())

            correct_function_name = extract_correct_function_name(test_code_str)
            if correct_function_name != "":
                collected_answer = rename_function(collected_answer, correct_function_name)

            test_block = extract_python_code(test_code_str.strip())
            test_list = [test_block[0] + "\n\n" + block for block in test_block[1:]]

            report_lines = []
            success_examples = 0

            for test in test_list:
                func_name_match = re.search(r"def\s+(test_\w+)\s*\(", test)
                func_name = func_name_match.group(1) if func_name_match else "unknown_test"

                is_passing, feedback, _ = self.execute(collected_answer, [test], timeout=timeout)

                if is_passing:
                    success_examples += 1
                    report_lines.append(f"PASS Test passed for '{func_name}'")
                else:
                    report_lines.append(f"FAIL Test failed for '{func_name}': \n{extract_failed_tests(feedback)}")
            
            if len(test_list) != 0:
                avg_score = success_examples / len(test_list)
            else:
                avg_score = 1.0 
            avg_scores.append(avg_score)  
            report_lines.append(f"\nAverage correctness: {avg_score:.2f}")

            reports.append("\n".join(report_lines))

        return reports, avg_scores

        
