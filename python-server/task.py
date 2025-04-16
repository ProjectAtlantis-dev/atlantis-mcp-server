#!/usr/bin/env python3
import logging
import json
from typing import Any, Dict, List, Optional, Union
from mcp.types import TextContent, Annotations

# Import shared state
from state import logger, next_task_id

async def task_add(args: dict) -> list[TextContent]:
    """Adds a new task using the provided payload."""
    logger.info(f"⚙️ TASK ADD CALLED with args: {args}")
    try:
        # Extract the payload from the arguments
        task_payload = args.get('payload')
        if task_payload is None:
            raise ValueError("Missing 'payload' in arguments")
        if not isinstance(task_payload, dict):
            raise ValueError("'payload' must be a JSON object (dictionary)")

        # Generate a new task ID
        global next_task_id
        task_id = next_task_id
        next_task_id += 1

        # Store the task details (the extracted payload dictionary)
        tasks[task_id] = task_payload  # Store the payload, not the whole args

        logger.info(f"✅ Task added with ID: {task_id}, Details: {task_payload}")
        # Return the new task ID as string in 'text' and int in annotations.task_id_int
        return [
            TextContent(
                type="text",
                text=str(task_id),
                annotations=Annotations(task_id_int=task_id)  # Add ID to annotations
            )
        ]
    except Exception as e:
        logger.error(f"❌ Error adding task: {str(e)}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        raise e

async def task_run(args: dict) -> list[TextContent]:
    """Runs a dynamic Python function with the task data."""
    logger.info(f"🏃 TASK RUN CALLED with args: {args}")
    task_id_str = args.get('id')
    if task_id_str is None:
        raise ValueError("Missing 'id' in arguments")

    try:
        task_id = int(task_id_str)
    except ValueError:
        raise ValueError("'id' must be an integer")

    # Retrieve the task data
    task_data = tasks.get(task_id)
    if task_data is None:
        raise ValueError(f"Task ID {task_id} not found")

    # Extract payload (handle potential prior result storage)
    payload = task_data.get('payload', task_data)  # Use get for dicts
    if not isinstance(payload, dict):
         raise ValueError(f"Task ID {task_id} does not contain a valid payload object.")

    function_name = payload.get('functionName')
    function_args = payload.get('arguments')  # MCP spec uses 'arguments'

    if not isinstance(function_name, str):
         raise ValueError(f"Task ID {task_id} payload missing 'functionName' string.")
    if function_args is None:  # Check for explicit None
         raise ValueError(f"Task ID {task_id} payload missing 'arguments'.")

    # Find the dynamic function definition
    dynamic_func = dynamic_functions.get(function_name)
    if dynamic_func is None or not callable(dynamic_func):
         raise ValueError(f"Dynamic function '{function_name}' not found or not callable.")

    logger.info(f"🚀 Executing dynamic function '{function_name}' for task {task_id} with args: {function_args}")

    # Execute the dynamic function
    try:
        result = await dynamic_func(function_args)
        # Assuming result is already in a format suitable for storage (e.g., list[TextContent])
        # Or convert if necessary. For now, store raw result.
        logger.info(f"✅ Dynamic function '{function_name}' for task {task_id} completed.")
    except Exception as e:
        logger.error(f"❌ Error executing dynamic function '{function_name}' for task {task_id}: {e}", exc_info=True)
        raise  # Re-raise the exception after logging

    # Store the result back with the task data
    # Preserve original payload, add/update result
    tasks[task_id] = {'payload': payload, 'result': result}

    return [TextContent(type="text", text=f"Task {task_id} executed successfully, result stored.")]

async def task_remove(args: dict) -> list[TextContent]:
    """Removes a task by its ID."""
    logger.info(f"🗑️ TASK REMOVE CALLED with args: {args}")
    task_id_str = args.get('id')
    if task_id_str is None:
        raise ValueError("Missing 'id' in arguments")

    try:
        task_id = int(task_id_str)
    except ValueError:
        raise ValueError("'id' must be an integer")

    # Attempt to remove the task from the dictionary
    if task_id in tasks:
        del tasks[task_id]
        logger.info(f"✅ Task {task_id} removed successfully.")
        return [TextContent(type="text", text=f"Task {task_id} removed successfully.")]
    else:
        logger.warning(f"❓ Task ID {task_id} not found for removal.")
        # Raise error as the task didn't exist
        raise ValueError(f"Task ID {task_id} not found")

async def task_peek(args: dict) -> list[TextContent]:
    """Retrieve the stored details for a specific task ID."""
    logger.info(f"👀 TASK PEEK CALLED with args: {args}")
    task_id_str = args.get('id')
    if task_id_str is None:
        raise ValueError("Missing 'id' in arguments")

    try:
        task_id = int(task_id_str)  # Ensure ID is an integer
    except ValueError:
        raise ValueError("'id' must be an integer")

    # Retrieve the task details from the dictionary
    task_details = tasks.get(task_id)

    if task_details is not None:
        logger.info(f"✅ Task {task_id} details found: {task_details}")
        # Return the stored task details as JSON string in 'text' and raw dict in annotations.task_payload_json
        return [
            TextContent(
                type="text",
                text=json.dumps(task_details),
                annotations=Annotations(task_payload_json=task_details)  # Add payload to annotations
            )
        ]
    else:
        logger.warning(f"❓ Task ID {task_id} not found.")
        raise ValueError(f"Task ID {task_id} not found")
