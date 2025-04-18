# Example dynamic function demonstrating client logging
def barf(foo:str, count:int=3):
    """
    Demonstrates client-side logging capabilities.
    
    Args:
        foo: A string to barf out
        count: Number of times to repeat (default: 3)
    
    Returns:
        Dictionary with result information
    """
    # Debug level log with function start info
    client_log(f"Starting barf function with arguments: foo='{foo}', count={count}", level="debug")
    
    # Regular info log
    client_log("Processing barf request...", logger_name="barf_processor")
    
    # Some mock processing
    if not isinstance(foo, str):
        error_msg = f"Expected string for 'foo', got {type(foo).__name__}"
        client_log(error_msg, level="error")
        return {"status": "error", "message": error_msg}
    
    if count <= 0:
        warn_msg = "Count must be positive, using default of 1"
        client_log(warn_msg, level="warning")
        count = 1
    
    # Generate the barfed output
    result = foo * count
    
    # Log structured data
    client_log({
        "status": "success",
        "input_length": len(foo),
        "output_length": len(result),
        "repetition_count": count
    }, level="info")
    
    # Return result
    return {
        "status": "success",
        "original": foo,
        "count": count,
        "result": result
    }
