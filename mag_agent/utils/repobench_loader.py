"""
RepoBench dataset loader module.

This module provides functionality to load and process RepoBench tasks for code completion evaluation.
RepoBench is a dataset focused on repository-level code completion tasks.
"""

import json
import os
import glob
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from mag_agent.utils.logging_config import logger


@dataclass
class RepoBenchTask:
    """A structured representation of a single RepoBench task instance."""
    task_id: str
    cropped_code: str
    context: str
    next_line: str
    repo_name: str
    file_path: str
    raw_data: Dict[str, Any]


def load_repobench_instance(file_path: str) -> RepoBenchTask:
    """
    Parses one RepoBench task JSON file.

    Args:
        file_path (str): The path to the RepoBench .json task file.

    Returns:
        RepoBenchTask: A structured data object containing task details.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return RepoBenchTask(
        task_id=data.get("task_id", os.path.basename(file_path)),
        cropped_code=data.get("cropped_code", ""),
        context=data.get("context", ""),
        next_line=data.get("next_line", ""),
        repo_name=data.get("repo_name", ""),
        file_path=data.get("file_path", ""),
        raw_data=data
    )


def load_repobench_dataset(dataset_path: str) -> List[RepoBenchTask]:
    """
    Load all RepoBench tasks from a directory containing JSON files.
    
    Args:
        dataset_path (str): Path to directory containing RepoBench JSON files
        
    Returns:
        List[RepoBenchTask]: List of parsed RepoBench tasks
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    json_files = glob.glob(os.path.join(dataset_path, "*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {dataset_path}")
    
    tasks = []
    failed_tasks = []
    
    for json_file in json_files:
        try:
            task = load_repobench_instance(json_file)
            tasks.append(task)
            
        except Exception as e:
            logger.error(f"Failed to load task from {json_file}: {e}")
            failed_tasks.append(json_file)
    
    if failed_tasks:
        logger.warning(f"Failed to load {len(failed_tasks)} tasks: {failed_tasks}")
    
    logger.info(f"Successfully loaded {len(tasks)} RepoBench tasks from {dataset_path}")
    return tasks


class RepoBenchLoader:
    """
    A loader class for RepoBench datasets that provides additional functionality
    for task filtering, sampling, and metadata extraction.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the RepoBench loader.
        
        Args:
            dataset_path (str): Path to the RepoBench dataset directory
        """
        self.dataset_path = dataset_path
        self.tasks = None
        self._metadata = None
    
    def load_tasks(self, max_tasks: Optional[int] = None) -> List[RepoBenchTask]:
        """
        Load tasks from the dataset with optional limit.
        
        Args:
            max_tasks (Optional[int]): Maximum number of tasks to load
            
        Returns:
            List[RepoBenchTask]: List of loaded tasks
        """
        if self.tasks is None:
            self.tasks = load_repobench_dataset(self.dataset_path)
        
        if max_tasks is not None and max_tasks > 0:
            return self.tasks[:max_tasks]
        
        return self.tasks
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the loaded dataset.
        
        Returns:
            Dict[str, Any]: Dataset metadata
        """
        if self.tasks is None:
            self.load_tasks()
        
        if self._metadata is None:
            repo_names = set(task.repo_name for task in self.tasks if task.repo_name)
            file_extensions = set(
                os.path.splitext(task.file_path)[1] 
                for task in self.tasks 
                if task.file_path and '.' in task.file_path
            )
            
            self._metadata = {
                "total_tasks": len(self.tasks),
                "unique_repositories": len(repo_names),
                "repository_names": sorted(list(repo_names)),
                "file_extensions": sorted(list(file_extensions)),
                "dataset_path": self.dataset_path
            }
        
        return self._metadata
    
    def filter_by_repo(self, repo_name: str) -> List[RepoBenchTask]:
        """
        Filter tasks by repository name.
        
        Args:
            repo_name (str): Repository name to filter by
            
        Returns:
            List[RepoBenchTask]: Filtered tasks
        """
        if self.tasks is None:
            self.load_tasks()
        
        return [task for task in self.tasks if task.repo_name == repo_name]
    
    def filter_by_extension(self, extension: str) -> List[RepoBenchTask]:
        """
        Filter tasks by file extension.
        
        Args:
            extension (str): File extension to filter by (with or without dot)
            
        Returns:
            List[RepoBenchTask]: Filtered tasks
        """
        if self.tasks is None:
            self.load_tasks()
        
        if not extension.startswith('.'):
            extension = '.' + extension
            
        return [
            task for task in self.tasks 
            if task.file_path and task.file_path.endswith(extension)
        ]


def create_repobench_prompt(task: RepoBenchTask) -> str:
    """
    Create a formatted prompt for RepoBench code completion task.
    
    Args:
        task (RepoBenchTask): RepoBench task instance
        
    Returns:
        str: Formatted prompt for the LLM
    """
    prompt = f"""You are an expert programmer working on code completion. Based on the provided code context and the cropped code snippet, predict the next line of code.

Repository: {task.repo_name}
File: {task.file_path}

Context:
{task.context}

Code snippet (with missing next line):
{task.cropped_code}

Please provide only the next line of code that should follow. Do not provide explanations or multiple alternatives - just the single most likely next line."""

    return prompt


def evaluate_repobench_prediction(predicted_line: str, actual_line: str) -> bool:
    """
    Evaluate whether the predicted line matches the actual next line.
    
    Args:
        predicted_line (str): The line predicted by the model
        actual_line (str): The actual next line from the dataset
        
    Returns:
        bool: True if the prediction is correct (exact match after stripping whitespace)
    """
    # Strip whitespace and compare
    predicted_clean = predicted_line.strip()
    actual_clean = actual_line.strip()
    
    return predicted_clean == actual_clean