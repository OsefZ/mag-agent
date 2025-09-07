import litellm
from typing import Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

class LLMError(Exception):
    """Exception raised for LLM API errors."""
    pass
from mag_agent.utils.logging_config import logger
from mag_agent.credit_tracker import get_credit_tracker

class LiteLLMClient:
    """Client for language model API interactions through LiteLLM.
    
    Features:
    - Automatic retry logic with exponential backoff
    - Credit usage tracking and budget enforcement
    - Support for multiple model providers
    """
    
    def __init__(self, enable_tracking: bool = True):
        """Initialize LLM client with optional credit tracking.
        
        Args:
            enable_tracking: Enable cost monitoring and budget limits
        """
        self.enable_tracking = enable_tracking
        self.credit_tracker = get_credit_tracker() if enable_tracking else None

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def generate(
        self,
        model: str,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        skip_tracking: bool = False
    ) -> Dict[str, Any]:
        """Generate text completion using specified language model.

        Args:
            model: Model identifier (e.g., "openai/gpt-4o-mini")
            messages: Conversation messages in OpenAI format
            temperature: Sampling randomness (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            skip_tracking: Bypass cost tracking for this request

        Returns:
            API response containing generated text and metadata

        Raises:
            LLMError: When API request fails after retries
            Exception: When budget limit would be exceeded
        """
        # Verify budget before API request
        if self.enable_tracking and not skip_tracking and self.credit_tracker:
            # Conservative cost estimation using token counts
            estimated_tokens = len(str(messages)) // 4 + max_tokens
            input_price, output_price = self.credit_tracker.get_model_pricing(model)
            estimated_cost = (estimated_tokens / 1_000_000) * max(input_price, output_price)
            
            can_proceed, message = self.credit_tracker.check_budget(estimated_cost)
            if not can_proceed:
                logger.error(message)
                raise Exception(f"Credit limit exceeded: {message}")
            
            logger.info(f"Budget check passed: {message}")
        
        logger.debug(f"Sending request to LiteLLM for model '{model}'...")
        
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            logger.debug(f"Received successful response from LiteLLM.")
            
            # Track usage after successful call
            if self.enable_tracking and not skip_tracking and self.credit_tracker:
                try:
                    self.credit_tracker.track_usage(
                        model=model,
                        messages=messages,
                        response=response
                    )
                except Exception as e:
                    logger.error(f"Failed to track usage: {e}")
                    # If we've hit the hard cap, raise the exception
                    if "HARD CAP" in str(e):
                        raise
            
            return response
            
        except Exception as e:
            logger.error(f"LiteLLM request failed after retries: {e}")
            raise LLMError(f"LiteLLM request failed: {e}")