import psutil
import os
import shutil
import config
import GPUtil
import time

GB = 1024 * 1024 * 1024


def get_detailed_system_performance():
    """
    Returns detailed performance data about the system, including CPU, memory, disk, and network utilization.

    Returns:
        dict: A dictionary containing detailed system performance metrics.
    """
    # CPU Utilization
    cpu_utilization = psutil.cpu_percent(
        interval=0.01, percpu=True
    )  # Percentage for each CPU
    cpu_freq = psutil.cpu_freq(percpu=True)  # Frequency for each CPU

    # Memory Utilization
    memory = psutil.virtual_memory()
    memory_utilization = {
        "total": memory.total / GB,
        "available": memory.available / GB,
        "percent": memory.percent,
        "used": memory.used / GB,
        "free": memory.free / GB,
    }
    # Disk Utilization
    storage = {
        "used": get_directory_size(config.data_dir),
        "free": get_partition_free_space(config.data_dir),
    }

    return {
        "cpu": {"utilization": cpu_utilization, "frequency": cpu_freq},
        "memory": memory_utilization,
        "storage": storage,
        "gpu": get_gpu_info(),
    }


def get_directory_size(directory):
    """
    Calculate the total size of all files in the specified directory.

    Args:
    directory (str): Path to the directory.

    Returns:
    float: Total size of all files in the directory in GB.
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # Skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size / GB


def get_partition_free_space(directory):
    """
    Get the free space of the partition in which the specified directory resides.

    Args:
    directory (str): Path to the directory.

    Returns:
    float: Free space of the partition in GB.
    """
    total, used, free = shutil.disk_usage(directory)
    if config.max_data_dir_size_gb > 0:
        return config.max_data_dir_size_gb - get_directory_size(directory)
    else:
        return free / GB


def get_gpu_info():
    """
    Get information about all available NVIDIA GPUs using GPUtil.

    Returns:
        list: A list of dictionaries, each containing details about a GPU.
    """
    gpus = GPUtil.getGPUs()
    gpu_info = []

    for gpu in gpus:
        info = {
            "id": gpu.id,
            "name": gpu.name,
            "load": gpu.load,
            "free_memory": gpu.memoryFree,
            "used_memory": gpu.memoryUsed,
            "total_memory": gpu.memoryTotal,
            "temperature": gpu.temperature,
        }
        gpu_info.append(info)

    return gpu_info
