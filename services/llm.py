import os
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator, Union
import logging
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI
from lib.retry_llm_call import retry_llm_call

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)

# Get API keys from environment
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

@retry_llm_call()
async def stream_text(
    prompt: str,
    model: str = "google/gemini-3-pro-preview",
    max_tokens: int = 4096,
    system_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    callback: Optional[Callable] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream text responses from OpenRouter API asynchronously.

    Args:
        prompt: The user prompt to send to the model.
        model: The model identifier on OpenRouter.
        max_tokens: Maximum number of tokens in the response.
        system_prompt: Optional system prompt.
        messages: Optional list of message objects (overrides prompt if provided).
        callback: Optional async callback function to process streaming events.
        response_format: Optional dictionary for structured outputs.

    Yields:
        Dictionary containing event information for each streaming event.
    """
    logger.info(f"Starting async stream_text with model: {model}")
    
    # Ensure event loop has a default ThreadPoolExecutor
    try:
        loop = asyncio.get_running_loop()
        if getattr(loop, "_default_executor", None) is None:
            import concurrent.futures
            loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=4))
            logger.debug("Set new default ThreadPoolExecutor for event loop")
    except RuntimeError:
        pass

    # Check if executor has been shut down
    try:
        loop = asyncio.get_running_loop()
        executor = getattr(loop, "_default_executor", None)
        if executor and hasattr(executor, "_shutdown") and executor._shutdown:
            import concurrent.futures
            new_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
            loop.set_default_executor(new_executor)
            logger.debug("Replaced shut down ThreadPoolExecutor with new one")
    except Exception as e:
        logger.debug(f"Could not check/replace executor: {e}")

    try:
        if not OPENROUTER_API_KEY:
            logger.error("OPENROUTER_API_KEY is not configured")
            raise ValueError("OPENROUTER_API_KEY is not configured")

        logger.debug("Initializing AsyncOpenAI client for OpenRouter")
        client = AsyncOpenAI(
            base_url=OPENROUTER_API_BASE,
            api_key=OPENROUTER_API_KEY,
        )

        # Configure messages
        if messages is None:
            logger.debug("Using single prompt message")
            messages_config = [{"role": "user", "content": prompt}]
        else:
            logger.debug(f"Using provided messages array with {len(messages)} messages")
            messages_config = messages

        # Add system prompt if provided
        if system_prompt and not any(msg['role'] == 'system' for msg in messages_config):
            logger.debug("Prepending system prompt")
            messages_config.insert(0, {"role": "system", "content": system_prompt})

        # Prepare stream parameters
        stream_params: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages_config,
            "stream": True,
        }
        
        if response_format:
            stream_params["response_format"] = response_format
            logger.info(f"Using structured outputs with format: {response_format.get('type')}")

        logger.info("Starting async stream with OpenRouter API")
        
        stream = await client.chat.completions.create(**stream_params)
        logger.info("Stream connection established")
        
        async for chunk in stream:
            # logger.debug(f"Received chunk: {chunk.id}") # Reduce noise
            
            # Log finish reason if present
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'finish_reason') and choice.finish_reason:
                    logger.info(f"Stream chunk finish_reason: {choice.finish_reason}")
                
                # Optional: Log content for debugging (can be verbose)
                # if hasattr(choice, 'delta') and hasattr(choice.delta, 'content') and choice.delta.content:
                #     logger.debug(f"Chunk content: {choice.delta.content}")

            # Process the event with callback
            if callback:
                logger.debug("Calling user-provided callback")
                await callback(chunk)

            # Yield the chunk to the caller
            yield chunk

        logger.info("Stream completed successfully")

    except Exception as e:
        logger.error(f"Error in async stream_text: {str(e)}", exc_info=True)
        raise
