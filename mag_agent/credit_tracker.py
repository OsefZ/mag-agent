"""
Credit tracking system for monitoring LLM API usage and enforcing spending limits.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import litellm

from mag_agent.utils.logging_config import logger

# Simple custom exception class since we deleted exceptions.py
class ConfigurationError(Exception):
    """Configuration error exception."""
    pass


@dataclass
class UsageRecord:
    """Record of a single LLM API call."""
    timestamp: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
    messages_preview: str
    response_preview: str


class CreditTracker:
    """
    Tracks LLM API usage and enforces spending limits.
    
    Features:
    - Tracks token usage per API call
    - Calculates costs based on model pricing
    - Enforces hard spending cap
    - Persists usage logs to disk
    - Provides usage summaries
    """
    
    # Model pricing per 1M tokens (as of 2024)
    # Format: (input_price, output_price) per 1M tokens
    MODEL_PRICING = {
        "gpt-4": (30.0, 60.0),
        "gpt-4-turbo": (10.0, 30.0),
        "gpt-4o": (5.0, 15.0),
        "gpt-4o-mini": (0.15, 0.6),
        "gpt-3.5-turbo": (0.5, 1.5),
        "claude-3-opus": (15.0, 75.0),
        "claude-3-sonnet": (3.0, 15.0),
        "claude-3-haiku": (0.25, 1.25),
        "claude-2.1": (8.0, 24.0),
        "claude-2": (8.0, 24.0),
        # Default pricing for unknown models
        "default": (10.0, 30.0)
    }
    
    def __init__(self, 
                 hard_cap: float = 30.0,
                 warning_threshold: float = 0.9,
                 usage_log_dir: str = "usage_logs"):
        """
        Initialize the credit tracker.
        
        Args:
            hard_cap: Maximum spending limit in USD (default: $30)
            warning_threshold: Fraction of hard_cap to trigger warnings (0.9 = 90%)
            usage_log_dir: Directory to store usage logs
        """
        self.hard_cap = hard_cap
        self.warning_threshold = warning_threshold * hard_cap
        self.usage_log_dir = Path(usage_log_dir)
        
        # Create usage log directory
        self.usage_log_dir.mkdir(exist_ok=True)
        
        # Load existing usage data
        self.total_spent = self._load_total_spent()
        
        # Current session file
        self.session_file = self._create_session_file()
        
        logger.info(f"Credit tracker initialized. Total spent so far: ${self.total_spent:.4f}")
        logger.info(f"Hard cap: ${self.hard_cap}, Warning at: ${self.warning_threshold:.2f}")
    
    def _create_session_file(self) -> Path:
        """Create a new session log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_file = self.usage_log_dir / f"session_{timestamp}.jsonl"
        
        # Write session header
        with open(session_file, 'w') as f:
            header = {
                "type": "session_start",
                "timestamp": datetime.now().isoformat(),
                "hard_cap": self.hard_cap,
                "starting_balance": self.total_spent
            }
            f.write(json.dumps(header) + "\n")
        
        return session_file
    
    def _load_total_spent(self) -> float:
        """Load total spending from all previous sessions."""
        total = 0.0
        
        if not self.usage_log_dir.exists():
            return total
        
        # Sum up all session totals
        for session_file in self.usage_log_dir.glob("session_*.jsonl"):
            try:
                with open(session_file, 'r') as f:
                    for line in f:
                        record = json.loads(line)
                        if record.get("type") == "usage":
                            total += record.get("estimated_cost", 0)
            except Exception as e:
                logger.warning(f"Error reading session file {session_file}: {e}")
        
        # Also check for summary file
        summary_file = self.usage_log_dir / "usage_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    total = max(total, summary.get("total_spent", 0))
            except Exception as e:
                logger.warning(f"Error reading summary file: {e}")
        
        return total
    
    def check_budget(self, estimated_cost: float = 0.5) -> Tuple[bool, str]:
        """
        Check if we have budget for an operation.
        
        Args:
            estimated_cost: Estimated cost of the operation in USD
        
        Returns:
            Tuple of (can_proceed, message)
        """
        remaining = self.hard_cap - self.total_spent
        
        if self.total_spent >= self.hard_cap:
            return False, f"HARD CAP REACHED! Spent ${self.total_spent:.4f} of ${self.hard_cap} limit"
        
        if self.total_spent + estimated_cost > self.hard_cap:
            return False, f"Operation would exceed hard cap! Current: ${self.total_spent:.4f}, Estimated: ${estimated_cost:.4f}, Limit: ${self.hard_cap}"
        
        if self.total_spent >= self.warning_threshold:
            logger.warning(f"APPROACHING LIMIT! Spent ${self.total_spent:.4f} of ${self.hard_cap} (${remaining:.4f} remaining)")
        
        return True, f"Budget OK: ${remaining:.4f} remaining of ${self.hard_cap} limit"
    
    def get_model_pricing(self, model: str) -> Tuple[float, float]:
        """
        Get pricing for a model.
        
        Returns:
            Tuple of (input_price_per_1M, output_price_per_1M)
        """
        # Check for exact match
        for key, pricing in self.MODEL_PRICING.items():
            if key in model.lower():
                return pricing
        
        # Use default pricing
        logger.warning(f"Unknown model '{model}', using default pricing")
        return self.MODEL_PRICING["default"]
    
    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate the cost of an API call.
        
        Args:
            model: Model name
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
        
        Returns:
            Estimated cost in USD
        """
        input_price, output_price = self.get_model_pricing(model)
        
        # Calculate cost (prices are per 1M tokens)
        input_cost = (prompt_tokens / 1_000_000) * input_price
        output_cost = (completion_tokens / 1_000_000) * output_price
        
        return input_cost + output_cost
    
    def track_usage(self, 
                   model: str,
                   messages: list,
                   response: Dict,
                   prompt_tokens: Optional[int] = None,
                   completion_tokens: Optional[int] = None) -> UsageRecord:
        """
        Track usage from an LLM API call.
        
        Args:
            model: Model name used
            messages: Input messages
            response: Response from LiteLLM
            prompt_tokens: Override for prompt tokens (if not in response)
            completion_tokens: Override for completion tokens (if not in response)
        
        Returns:
            UsageRecord of the tracked usage
        
        Raises:
            Exception if hard cap is exceeded
        """
        # Extract token counts
        usage = response.get("usage", {})
        prompt_tokens = prompt_tokens or usage.get("prompt_tokens", 0)
        completion_tokens = completion_tokens or usage.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens
        
        # Calculate cost
        cost = self.calculate_cost(model, prompt_tokens, completion_tokens)
        
        # Check if this would exceed the hard cap
        if self.total_spent + cost > self.hard_cap:
            error_msg = f"HARD CAP EXCEEDED! Current: ${self.total_spent:.4f} + New: ${cost:.4f} = ${self.total_spent + cost:.4f} > Limit: ${self.hard_cap}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        # Create usage record
        record = UsageRecord(
            timestamp=datetime.now().isoformat(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=cost,
            messages_preview=str(messages[0] if messages else "")[:100],
            response_preview=str(response.get("choices", [{}])[0].get("message", {}).get("content", ""))[:100]
        )
        
        # Update total
        self.total_spent += cost
        
        # Log to file
        with open(self.session_file, 'a') as f:
            usage_entry = {
                "type": "usage",
                **asdict(record)
            }
            f.write(json.dumps(usage_entry) + "\n")
        
        # Update summary file
        self._update_summary()
        
        # Check warnings
        if self.total_spent >= self.warning_threshold:
            logger.warning(f"WARNING: Approaching spending limit! Spent ${self.total_spent:.4f} of ${self.hard_cap}")
        
        logger.info(f"Tracked usage: {total_tokens} tokens, ${cost:.4f} cost. Total spent: ${self.total_spent:.4f}")
        
        return record
    
    def _update_summary(self):
        """Update the summary file with current totals."""
        summary_file = self.usage_log_dir / "usage_summary.json"
        
        summary = {
            "last_updated": datetime.now().isoformat(),
            "total_spent": self.total_spent,
            "hard_cap": self.hard_cap,
            "remaining": self.hard_cap - self.total_spent,
            "percentage_used": (self.total_spent / self.hard_cap) * 100 if self.hard_cap > 0 else 0
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def get_usage_summary(self) -> Dict:
        """
        Get a summary of current usage.
        
        Returns:
            Dictionary with usage statistics
        """
        return {
            "total_spent": self.total_spent,
            "hard_cap": self.hard_cap,
            "remaining": self.hard_cap - self.total_spent,
            "percentage_used": (self.total_spent / self.hard_cap) * 100 if self.hard_cap > 0 else 0,
            "at_warning": self.total_spent >= self.warning_threshold,
            "at_limit": self.total_spent >= self.hard_cap
        }
    
    def reset_tracking(self, confirm: bool = False):
        """
        Reset all tracking data (use with caution).
        
        Args:
            confirm: Must be True to actually reset
        """
        if not confirm:
            logger.warning("Reset tracking called without confirmation. Set confirm=True to reset.")
            return
        
        logger.warning("Resetting all usage tracking data!")
        
        # Archive current logs
        archive_dir = self.usage_log_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for file in self.usage_log_dir.glob("*.json*"):
            if file.is_file():
                file.rename(archive_dir / f"{timestamp}_{file.name}")
        
        # Reset totals
        self.total_spent = 0.0
        self.session_file = self._create_session_file()
        self._update_summary()
        
        logger.info("Usage tracking reset complete")


# Global instance for easy access
_credit_tracker = None

def get_credit_tracker() -> CreditTracker:
    """Get or create the global credit tracker instance."""
    global _credit_tracker
    if _credit_tracker is None:
        _credit_tracker = CreditTracker()
    return _credit_tracker