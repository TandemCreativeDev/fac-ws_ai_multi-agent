import os
import re
import datetime
from typing import Dict,  Any


def extract_code_from_response(response_text: str) -> str:
    if not response_text:
        return ""

    code_block_pattern = r'```(?:python)?\s*(.*?)\s*```'
    match = re.search(code_block_pattern, response_text, re.DOTALL)
    return match.group(1).strip() if match else response_text.strip()


def sanitise_filename(text: str) -> str:
    sanitised = re.sub(r'[^\w\s-]', '', text).strip()
    return re.sub(r'[-\s]+', '_', sanitised).lower()


class CodebaseGenerator:
    def __init__(self, pattern_name: str, task: str):
        self.pattern_name = pattern_name
        self.task = task
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.folder_name = f"generated/{pattern_name}_{self.timestamp}"

    def create_folder(self) -> str:
        os.makedirs(self.folder_name, exist_ok=True)
        return self.folder_name

    def write_python_file(self, filename: str, content: str) -> None:
        code = extract_code_from_response(content)
        if code:
            filepath = os.path.join(self.folder_name, f"{filename}.py")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)

    def write_text_file(self, filename: str, content: str) -> None:
        filepath = os.path.join(self.folder_name, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)


class SequentialCodebase(CodebaseGenerator):
    def generate(self, result: Dict[str, Any]) -> None:
        self.create_folder()

        self.write_python_file("original_code", result.get('code', ''))
        self.write_python_file(
            "refactored_code", result.get('refactored_code', ''))

        if result.get('unit_tests'):
            self.write_python_file("unit_tests", result.get('unit_tests', ''))

        unit_tests_section = ""
        if result.get('unit_tests'):
            unit_tests_section = f"""

## Unit Tests
```python
{extract_code_from_response(result.get('unit_tests', 'No unit tests generated'))}
```"""

        files_generated = "- `original_code.py` - Initial implementation\n- `refactored_code.py` - Improved version based on review"
        if result.get('unit_tests'):
            files_generated += "\n- `unit_tests.py` - Comprehensive test suite"

        audit_content = f"""# Sequential Workflow Audit Trail

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Task:** {self.task}  
**Pattern:** Sequential Workflow

## Original Code
```python
{extract_code_from_response(result.get('code', 'No code generated'))}
```

## Review Feedback
{result.get('review', 'No review available')}

## Refactored Code
```python
{extract_code_from_response(result.get('refactored_code', 'No refactored code available'))}
```{unit_tests_section}

## Files Generated
{files_generated}

---
*Generated using LangGraph Sequential Workflow Pattern*
"""
        self.write_text_file("AUDIT_TRAIL.md", audit_content)
        print(f"✅ Sequential codebase created in: {self.folder_name}/")


class ConditionalCodebase(CodebaseGenerator):
    def generate(self, result: Dict[str, Any]) -> None:
        self.create_folder()

        code_list = result.get('code', [''])
        if isinstance(code_list, str):
            code_list = [code_list]

        final_code = code_list[-1] if code_list else ''
        self.write_python_file("final_code", final_code)

        previous_iterations_section = ""
        if len(code_list) > 1:
            previous_iterations_section = "\n## Previous Iterations\n\n"
            for i, code_version in enumerate(code_list[:-1]):
                iteration_label = "Original Code" if i == 0 else f"Iteration {i}"
                previous_iterations_section += f"""### {iteration_label}
```python
{extract_code_from_response(code_version)}
```

"""

        # Enhanced multi-criteria scoring section
        quality_metrics_section = self._build_quality_metrics_section(result)

        audit_content = f"""# Conditional Routing Audit Trail

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Task:** {self.task}  
**Pattern:** Conditional Routing

## Final Code
```python
{extract_code_from_response(final_code)}
```

## Review Feedback
{result.get('review', 'No review available')}

{quality_metrics_section}{previous_iterations_section}
## Files Generated
- `final_code.py` - Quality-approved implementation

---
*Generated using LangGraph Conditional Routing Pattern*
"""
        self.write_text_file("AUDIT_TRAIL.md", audit_content)
        print(
            f"✅ Conditional routing codebase created in: {self.folder_name}/")

    def _build_quality_metrics_section(self, result: Dict[str, Any]) -> str:
        # Check if we have multi-criteria scores
        if all(key in result for key in ['security_score', 'performance_score', 'readability_score']):
            security = result['security_score']
            performance = result['performance_score']
            readability = result['readability_score']
            lowest = result.get('lowest_score', min(
                security, performance, readability))

            # Create visual score bars
            def score_bar(score: int) -> str:
                filled = "█" * score
                empty = "░" * (10 - score)
                return f"{filled}{empty} ({score}/10)"

            return f"""## Quality Metrics (Multi-Criteria Evaluation)

| Criterion    | Score | Visual                    | Status |
|--------------|-------|---------------------------|--------|
| Security     | {security}/10  | {score_bar(security)}     | {'✅ Pass' if security >= 7 else '❌ Fail'} |
| Performance  | {performance}/10  | {score_bar(performance)}  | {'✅ Pass' if performance >= 7 else '❌ Fail'} |
| Readability  | {readability}/10  | {score_bar(readability)}  | {'✅ Pass' if readability >= 7 else '❌ Fail'} |
| **Overall**  | **{lowest}/10** | **{score_bar(lowest)}** | **{'✅ All Pass' if lowest >= 7 else '❌ Needs Work'}** |

**Evaluation Method:** Lowest score determines overall quality  
**Iterations:** {result.get('iteration_count', 0)}  
**Threshold:** 7/10 minimum for all criteria  

"""
        else:
            # Fallback to single score display
            quality_score = result.get('quality_score', 'N/A')
            iteration_count = result.get('iteration_count', 0)

            return f"""## Quality Metrics
- **Score:** {quality_score}/10
- **Iterations:** {iteration_count}
- **Threshold:** 7/10

"""


class ParallelCodebase(CodebaseGenerator):
    def generate(self, result: Dict[str, Any]) -> None:
        self.create_folder()

        self.write_python_file("main_code", result.get('code', ''))

        synthesis_content = f"""# Code Analysis Synthesis Report

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Task:** {self.task}  
**Analysis Method:** Parallel Expert Review

## Executive Summary

{result.get('final_report', 'No synthesis report available')}

---
*Generated using LangGraph Parallel Processing Pattern*
"""
        self.write_text_file("SYNTHESIS_REPORT.md", synthesis_content)

        audit_content = f"""# Parallel Processing Audit Trail

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Task:** {self.task}  
**Pattern:** Parallel Processing

## Generated Code
```python
{extract_code_from_response(result.get('code', 'No code generated'))}
```

## Expert Analysis Reports

### Security Analysis
{result.get('security_analysis', 'No security analysis available')}

### Performance Analysis
{result.get('performance_analysis', 'No performance analysis available')}

### Style Analysis
{result.get('style_analysis', 'No style analysis available')}

## Files Generated
- `main_code.py` - Analysed implementation
- `SYNTHESIS_REPORT.md` - **KEY DELIVERABLE:** Aggregated expert recommendations

---
*Generated using LangGraph Parallel Processing Pattern*
"""
        self.write_text_file("AUDIT_TRAIL.md", audit_content)
        print(
            f"✅ Parallel processing codebase created in: {self.folder_name}/")
        print(f"📊 Key deliverable: {self.folder_name}/SYNTHESIS_REPORT.md")


class SupervisorCodebase(CodebaseGenerator):
    def generate(self, result: Dict[str, Any]) -> None:
        self.create_folder()

        self.write_python_file("main_code", result.get('code', ''))

        final_analysis_content = f"""# Expert Analysis & Recommendations

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Task:** {self.task}  
**Analysis Method:** Supervised Expert Consultation

## Executive Summary

{result.get('final_analysis', 'No final analysis available')}

## Expert Consultation Process

**Agents Consulted:** {', '.join(result.get('completed_agents', []))}

### Supervisor Decision Log
{result.get('supervisor_notes', 'No supervisor decisions recorded')}

---
*Generated using LangGraph Supervisor Agents Pattern*
"""
        self.write_text_file("EXPERT_ANALYSIS.md", final_analysis_content)

        completed_agents = result.get('completed_agents', [])
        reports_section = ""

        if result.get('security_report'):
            reports_section += f"### Security Expert Report\n{result['security_report']}\n\n"
        if result.get('performance_report'):
            reports_section += f"### Performance Expert Report\n{result['performance_report']}\n\n"
        if result.get('quality_report'):
            reports_section += f"### Quality Expert Report\n{result['quality_report']}\n\n"

        audit_content = f"""# Supervisor Agents Audit Trail

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Task:** {self.task}  
**Pattern:** Supervisor Agents
**Agents Consulted:** {', '.join(completed_agents)}

## Generated Code
```python
{extract_code_from_response(result.get('code', 'No code generated'))}
```

## Supervisor Decision Process
{result.get('supervisor_notes', 'No supervisor notes available')}

## Individual Expert Reports

{reports_section}## Files Generated
- `main_code.py` - Expert-reviewed implementation  
- `EXPERT_ANALYSIS.md` - **KEY DELIVERABLE:** Synthesised expert recommendations

---
*Generated using LangGraph Supervisor Agents Pattern*
"""
        self.write_text_file("AUDIT_TRAIL.md", audit_content)
        print(f"✅ Supervisor agents codebase created in: {self.folder_name}/")
        print(f"🎯 Key deliverable: {self.folder_name}/EXPERT_ANALYSIS.md")


class EvaluatorCodebase(CodebaseGenerator):
    def generate(self, result: Dict[str, Any]) -> None:
        self.create_folder()

        self.write_python_file("final_code", result.get(
            'final_code', result.get('code', '')))

        current_eval = result.get('current_evaluation', {})
        final_score = current_eval.get('score', 'N/A')
        final_feedback = current_eval.get('feedback', 'No feedback available')
        iteration_count = result.get('iteration_count', 0)

        history_section = f"""## Optimisation Summary
- **Total Iterations:** {iteration_count}
- **Final Quality Score:** {final_score}/10
- **Final Feedback:** {final_feedback}
- **Completion Reason:** {'Quality threshold reached' if final_score >= 8 else 'Max iterations reached' if iteration_count >= 3 else 'Evaluator determined completion'}

"""

        audit_content = f"""# Evaluator-Optimiser Audit Trail

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Task:** {self.task}  
**Pattern:** Evaluator-Optimiser
**Total Iterations:** {iteration_count}
**Final Score:** {final_score}/10

## Final Code
```python
{extract_code_from_response(result.get('final_code', result.get('code', 'No code generated')))}
```

{history_section}## Files Generated
- `final_code.py` - Iteratively optimised implementation

---
*Generated using LangGraph Evaluator-Optimiser Pattern*
"""
        self.write_text_file("AUDIT_TRAIL.md", audit_content)
        print(
            f"✅ Evaluator-optimiser codebase created in: {self.folder_name}/")


class ProductionCodebase(CodebaseGenerator):
    def generate(self, result: Dict[str, Any]) -> None:
        self.create_folder()

        self.write_python_file("production_code", result.get(
            'refactored_code', result.get('code', '')))

        metrics_section = f"""## Production Metrics
- **Session ID:** {result.get('session_id', 'N/A')}
- **Execution Time:** {result.get('execution_time', 0):.2f}s
- **Retry Count:** {result.get('retry_count', 0)}
- **Human Approval Required:** {result.get('human_approval_needed', False)}
"""

        error_section = ""
        if result.get('error_log'):
            error_section = f"""## Error Log
```
{result['error_log']}
```

"""

        audit_content = f"""# Production Ready Audit Trail

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Task:** {self.task}  
**Pattern:** Production Ready

## Production Code
```python
{extract_code_from_response(result.get('refactored_code', result.get('code', 'No code generated')))}
```

{metrics_section}

## Review Feedback
{result.get('review', 'No review available')}

{error_section}## Files Generated
- `production_code.py` - Production-ready implementation

---
*Generated using LangGraph Production Ready Pattern*
"""
        self.write_text_file("AUDIT_TRAIL.md", audit_content)
        print(f"✅ Production ready codebase created in: {self.folder_name}/")
