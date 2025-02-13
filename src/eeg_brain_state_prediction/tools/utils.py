import os

def set_thread_env(config) -> None:
    """Set environment variables for thread control with validation

    Args:
        config: Configuration object with n_threads attribute

    Raises:
        ConfigurationError: If thread configuration is invalid
    """

    thread_vars = [
        "OMP_NUM_THREADS", 
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS"
    ]
    
    try:
        for var in thread_vars:
            os.environ[var] = str(config.n_threads)
    except Exception as e:
        raise ConfigurationError(f"Failed to set thread environment variables: {str(e)}")