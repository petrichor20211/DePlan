import json
import yaml
from pathlib import Path
from typing import Any, List, Dict, Optional
from pydantic_core import to_jsonable_python
import re


def read_json_file(json_file: str, encoding="utf-8") -> List[Any]:
    if not Path(json_file).exists():
        raise FileNotFoundError(f"json_file: {json_file} not exist, return []")

    with open(json_file, "r", encoding=encoding) as fin:
        try:
            data = json.load(fin)
        except Exception:
            raise ValueError(f"read json file: {json_file} failed")
    return data

def write_json_file(json_file: str, data: list, encoding: str = None, indent: int = 4):
    folder_path = Path(json_file).parent
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

    with open(json_file, "w", encoding=encoding) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=indent, default=to_jsonable_python)

def read_yaml_file(yaml_file: str, encoding='utf-8') -> Dict[str, Any]:
    if not Path(yaml_file).exists():
        raise FileNotFoundError(f"yaml_file: {yaml_file} not exist, return empty dict")
    
    with open(yaml_file, "r", encoding=encoding) as f:
        try:
            data = yaml.safe_load(f)
        except Exception:
            raise ValueError(f"read yaml file: {yaml_file} failed")
    return data
    
def parse_code_block(text: str, lang: str = "python") -> Optional[str]:
    """Extracts the first code block of a given language from a markdown-formatted text."""
    pattern = rf"```{lang}\s*\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def parse_xml_tag(response: str, xml_tag: str) -> str:
    pattern = rf"<{xml_tag}>(.*?)</{xml_tag}>"
    match = re.search(pattern, response, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_domain_name(domain_pddl: str) -> str:
    """Extract domain name from PDDL domain file.
    
    Parses (define (domain name) ...) structure.
    
    Args:
        domain_pddl: PDDL domain file content
        
    Returns:
        Domain name string, or "domain" as fallback
    """
    if not domain_pddl:
        return "domain"
        
    # Match (define (domain domain-name) ...)
    match = re.search(r'\(define\s+\(domain\s+([^\)]+)\)', domain_pddl)
    if match:
        return match.group(1).strip()
        
    return "domain"


def parse_pddl_from_response(response: str) -> str:
    """Parse PDDL content from LLM response.
    
    Handles cases where LLM wraps PDDL in markdown code blocks.
    
    Args:
        response: Raw LLM response
        
    Returns:
        Extracted PDDL string
    """
    # Try to extract from markdown code block
    if "```" in response:
        # Find content between triple backticks
        parts = response.split("```")
        for i, part in enumerate(parts):
            # Look for PDDL content (skip language tags like "pddl")
            if i % 2 == 1:  # Odd indices are inside code blocks
                # Remove language identifier if present
                lines = part.strip().split('\n')
                if lines and lines[0].lower() in ['pddl', 'lisp', 'scheme']:
                    return '\n'.join(lines[1:])
                return part.strip()
                
    # Return as-is if no code block found
    return response.strip()