"""Domain file loader for PDDL environments.

Handles loading domain definitions, examples, and task lists.
"""

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .pddl_utils import postprocess


@dataclass
class DomainContext:
    """Example context for few-shot prompting."""
    nl: str
    pddl: str
    solution: str


@dataclass
class DomainData:
    """Complete domain data package."""
    name: str
    pddl: str
    nl: str
    context: Optional[DomainContext]
    task_files: list[tuple[str, str]]  # [(nl_file, pddl_file), ...]


class DomainLoader:
    """Loads PDDL domain files and task lists."""
    
    def __init__(self, domain_path: Path):
        """Initialize loader with domain directory path.
        
        Args:
            domain_path: Path to domain directory containing PDDL files
        """
        self.domain_path = domain_path
        
    def load(self) -> DomainData:
        """Load complete domain data.
        
        Returns:
            DomainData object with all domain files loaded
            
        Raises:
            FileNotFoundError: If domain.pddl not found
        """
        return DomainData(
            name=self.domain_path.name,
            pddl=self._load_domain_pddl(),
            nl=self._load_domain_nl(),
            context=self._load_context(),
            task_files=self._load_task_list(),
        )
        
    def _load_domain_pddl(self) -> str:
        """Load domain PDDL definition."""
        path = self.domain_path / "domain.pddl"
        if not path.exists():
            raise FileNotFoundError(f"Domain PDDL not found: {path}")
        return postprocess(path.read_text())
        
    def _load_domain_nl(self) -> str:
        """Load natural language domain description (optional)."""
        path = self.domain_path / "domain.nl"
        return postprocess(path.read_text()) if path.exists() else ""
        
    def _load_context(self) -> Optional[DomainContext]:
        """Load few-shot example context (optional)."""
        nl_path = self.domain_path / "p_example.nl"
        pddl_path = self.domain_path / "p_example.pddl"
        sol_path = self.domain_path / "p_example.sol"
        
        if not all(p.exists() for p in [nl_path, pddl_path, sol_path]):
            return None
            
        return DomainContext(
            nl=postprocess(nl_path.read_text()),
            pddl=postprocess(pddl_path.read_text()),
            solution=postprocess(sol_path.read_text()),
        )
        
    def _load_task_list(self) -> list[tuple[str, str]]:
        """Load list of task file pairs.
        
        Returns:
            Sorted list of (nl_filename, pddl_filename) tuples
        """
        tasks = []
        pattern = str(self.domain_path / "*.nl")
        
        for nl_path in glob.glob(pattern):
            filename = os.path.basename(nl_path)
            # Skip domain and example files
            if self._is_task_file(filename):
                pddl_file = filename.replace(".nl", ".pddl")
                if (self.domain_path / pddl_file).exists():
                    tasks.append((filename, pddl_file))
        
        return sorted(tasks)
        
    @staticmethod
    def _is_task_file(filename: str) -> bool:
        """Check if file is a task (not domain or example)."""
        return "domain" not in filename and "p_example" not in filename

