"""AST-based Python code parser for structural analysis.

Provides extraction of functions, classes, and imports from Python source
files for building graph-based code representations.
"""

import ast
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

class CodeParser:
    """Python source code analyzer using Abstract Syntax Trees.
    
    Extracts structural information including function definitions,
    class declarations, and import statements from Python files.
    """
    
    def __init__(self):
        pass
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze Python file structure using AST parsing.
        
        Args:
            file_path: Path to Python source file
            
        Returns:
            Dictionary containing extracted functions, classes, and imports
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            result = {
                'file_path': file_path,
                'functions': [],
                'classes': [],
                'imports': [],
                'content': content
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    result['functions'].append({
                        'name': node.name,
                        'lineno': node.lineno,
                        'file_path': file_path
                    })
                elif isinstance(node, ast.ClassDef):
                    result['classes'].append({
                        'name': node.name,
                        'lineno': node.lineno,
                        'file_path': file_path
                    })
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        result['imports'].append({
                            'module': alias.name,
                            'alias': alias.asname,
                            'lineno': node.lineno
                        })
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        result['imports'].append({
                            'module': f"{module}.{alias.name}" if module else alias.name,
                            'alias': alias.asname,
                            'lineno': node.lineno
                        })
            
            return result
            
        except Exception as e:
            return {
                'file_path': file_path,
                'error': str(e),
                'functions': [],
                'classes': [],
                'imports': [],
                'content': ''
            }
    
    def extract_function_code(self, file_path: str, function_name: str) -> Optional[str]:
        """Extract source code for a specific function by name.
        
        Args:
            file_path: Path to source file
            function_name: Name of function to extract
            
        Returns:
            Function source code or None if not found
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            lines = content.splitlines()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
                    return '\n'.join(lines[start_line:end_line])
            
            return None
            
        except Exception:
            return None