"""
Distributed training utilities supporting both torchrun and MPI launchers.

This module provides unified distributed training support for:
- torchrun (PyTorch Elastic): Standard PyTorch distributed launcher
- mpirun (OpenMPI): MPI-based launcher commonly used in HPC environments

Environment variable mappings:
- torchrun: LOCAL_RANK, RANK, WORLD_SIZE, TORCHELASTIC_RESTART_COUNT
- mpirun: OMPI_COMM_WORLD_LOCAL_RANK, OMPI_COMM_WORLD_RANK, OMPI_COMM_WORLD_SIZE
"""

import logging
import os
import socket
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

logger = logging.getLogger("distributed")

BACKEND = "nccl"


def _get_mpi_env() -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Get MPI environment variables.

    Returns:
        Tuple of (local_rank, global_rank, world_size) or (None, None, None) if not in MPI.
    """
    # OpenMPI environment variables
    local_rank = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
    global_rank = os.environ.get("OMPI_COMM_WORLD_RANK")
    world_size = os.environ.get("OMPI_COMM_WORLD_SIZE")

    if local_rank is not None and global_rank is not None and world_size is not None:
        return int(local_rank), int(global_rank), int(world_size)

    # MPICH environment variables (alternative MPI implementation)
    local_rank = os.environ.get("MPI_LOCALRANKID")
    global_rank = os.environ.get("PMI_RANK")
    world_size = os.environ.get("PMI_SIZE")

    if local_rank is not None and global_rank is not None and world_size is not None:
        return int(local_rank), int(global_rank), int(world_size)

    return None, None, None


def _get_torchrun_env() -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Get torchrun environment variables.

    Returns:
        Tuple of (local_rank, global_rank, world_size) or (None, None, None) if not in torchrun.
    """
    local_rank = os.environ.get("LOCAL_RANK")
    global_rank = os.environ.get("RANK")
    world_size = os.environ.get("WORLD_SIZE")

    if local_rank is not None:
        return (
            int(local_rank),
            int(global_rank) if global_rank else int(local_rank),
            int(world_size) if world_size else 1,
        )

    return None, None, None


def detect_distributed_env() -> Tuple[str, int, int, int]:
    """
    Detect the distributed training environment.

    Returns:
        Tuple of (launcher_type, local_rank, global_rank, world_size)
        launcher_type: "torchrun", "mpi", or "single"
    """
    # Check MPI first (more specific environment variables)
    local_rank, global_rank, world_size = _get_mpi_env()
    if local_rank is not None:
        return "mpi", local_rank, global_rank, world_size

    # Check torchrun
    local_rank, global_rank, world_size = _get_torchrun_env()
    if local_rank is not None:
        return "torchrun", local_rank, global_rank, world_size

    # Single GPU fallback
    return "single", 0, 0, 1


def setup_distributed_env_for_mpi():
    """
    Setup environment variables for MPI-based distributed training.

    This function sets LOCAL_RANK, RANK, and WORLD_SIZE environment variables
    from MPI environment variables so that PyTorch distributed can work correctly.

    Note: CUDA_VISIBLE_DEVICES should be set by the launch script (train_mpi.sh),
    not modified here. Changing CUDA_VISIBLE_DEVICES after CUDA initialization
    (which may happen during imports) has no effect and causes device mismatch.
    """
    local_rank, global_rank, world_size = _get_mpi_env()

    if local_rank is None:
        logger.warning("MPI environment not detected, skipping MPI env setup")
        return

    # Set PyTorch distributed environment variables
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(global_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # NOTE: Do NOT modify CUDA_VISIBLE_DEVICES here.
    # The launch script should set CUDA_VISIBLE_DEVICES=0,1,2,...,N-1
    # and each process uses local_rank to select its GPU via torch.cuda.set_device()

    logger.info(
        f"MPI environment setup: LOCAL_RANK={local_rank}, "
        f"RANK={global_rank}, WORLD_SIZE={world_size}"
    )


def get_master_addr_port() -> Tuple[str, int]:
    """
    Get master address and port for distributed initialization.

    For MPI, we need to determine the master node ourselves.
    For torchrun, these are already set in environment variables.

    Returns:
        Tuple of (master_addr, master_port)
    """
    # Check if already set (torchrun sets these)
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT")

    if master_addr and master_port:
        return master_addr, int(master_port)

    # For MPI, rank 0 broadcasts its hostname
    launcher_type, local_rank, global_rank, world_size = detect_distributed_env()

    if launcher_type == "mpi":
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD

            if global_rank == 0:
                master_addr = socket.gethostname()
            else:
                master_addr = None

            master_addr = comm.bcast(master_addr, root=0)
            master_port = 29500  # Default PyTorch distributed port

            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = str(master_port)

            return master_addr, master_port

        except ImportError:
            # Fallback: use hostname of current node
            master_addr = socket.gethostname()
            master_port = 29500
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = str(master_port)
            return master_addr, master_port

    # Single GPU fallback
    return "localhost", 29500


def init_distributed(backend: str = BACKEND) -> str:
    """
    Initialize distributed training environment.

    Automatically detects whether running under torchrun or mpirun
    and initializes accordingly.

    Args:
        backend: Distributed backend to use ("nccl" for GPU training)

    Returns:
        The launcher type that was detected ("torchrun", "mpi", or "single")
    """
    launcher_type, local_rank, global_rank, world_size = detect_distributed_env()

    if launcher_type == "single":
        logger.warning(
            "No distributed environment detected. Running in single GPU mode. "
            "Use torchrun or mpirun for multi-GPU training."
        )
        return launcher_type

    # Setup MPI environment if needed
    if launcher_type == "mpi":
        setup_distributed_env_for_mpi()

    # Get master address and port
    master_addr, master_port = get_master_addr_port()

    logger.info(
        f"Initializing distributed: launcher={launcher_type}, "
        f"rank={global_rank}/{world_size}, local_rank={local_rank}, "
        f"master={master_addr}:{master_port}"
    )

    # Set CUDA device before init
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        logger.info(
            f"CUDA setup: device_count={device_count}, "
            f"CUDA_VISIBLE_DEVICES={cuda_visible}, "
            f"setting device to local_rank={local_rank}"
        )

        if local_rank >= device_count:
            raise RuntimeError(
                f"local_rank ({local_rank}) >= device_count ({device_count}). "
                f"CUDA_VISIBLE_DEVICES={cuda_visible}. "
                f"Ensure CUDA_VISIBLE_DEVICES includes enough GPUs for all ranks."
            )

        torch.cuda.set_device(local_rank)

    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=global_rank,
        )

    return launcher_type


@lru_cache()
def get_rank() -> int:
    """Get the global rank of this process."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


@lru_cache()
def get_local_rank() -> int:
    """Get the local rank (GPU index) of this process."""
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank)

    # Fallback for MPI
    local_rank, _, _ = _get_mpi_env()
    if local_rank is not None:
        return local_rank

    return 0


@lru_cache()
def get_world_size() -> int:
    """Get the total number of processes in distributed training."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def visible_devices() -> List[int]:
    """Get list of visible CUDA device indices."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        return [int(d) for d in cuda_visible.split(",")]
    return list(range(torch.cuda.device_count()))


def set_device():
    """
    Set the CUDA device for this process based on local rank.

    This function handles both torchrun and MPI environments.
    """
    logger.info(f"torch.cuda.device_count: {torch.cuda.device_count()}")

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

    local_rank = get_local_rank()
    logger.info(f"local rank: {local_rank}")

    assert torch.cuda.is_available(), "CUDA is not available"

    device_count = torch.cuda.device_count()

    if device_count == 1:
        # Single GPU visible (common in MPI with CUDA_VISIBLE_DEVICES set per process)
        torch.cuda.set_device(0)
        return

    # Multiple GPUs visible
    assert 0 <= local_rank < device_count, (
        f"local_rank {local_rank} is out of range for {device_count} GPUs"
    )

    logger.info(f"Set cuda device to {local_rank}")
    torch.cuda.set_device(local_rank)


def avg_aggregate(metric: Union[float, int]) -> Union[float, int]:
    """
    Average a metric across all processes.

    Args:
        metric: The metric value to average

    Returns:
        The averaged metric value
    """
    if not dist.is_initialized():
        return metric

    buffer = torch.tensor([metric], dtype=torch.float32, device="cuda")
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    return buffer[0].item() / get_world_size()


def is_torchrun() -> bool:
    """Check if running under torchrun launcher."""
    return "TORCHELASTIC_RESTART_COUNT" in os.environ


def is_mpi() -> bool:
    """Check if running under MPI launcher."""
    return (
        "OMPI_COMM_WORLD_RANK" in os.environ or
        "PMI_RANK" in os.environ
    )


def is_distributed() -> bool:
    """Check if running in distributed mode (either torchrun or MPI)."""
    return is_torchrun() or is_mpi() or dist.is_initialized()


def get_node_count() -> int:
    """
    Get the number of nodes in distributed training.

    For MPI, uses mpi4py to gather hostnames and count unique nodes.
    For torchrun, infers from world_size and local_world_size.

    Returns:
        Number of nodes
    """
    if not is_distributed():
        return 1

    # Try MPI method first
    if is_mpi():
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            hostname = socket.gethostname()
            all_hostnames = comm.gather(hostname, root=0)

            if rank == 0:
                num_nodes = len(set(all_hostnames))
            else:
                num_nodes = None

            num_nodes = comm.bcast(num_nodes, root=0)
            return num_nodes

        except ImportError:
            pass

    # Fallback: try to infer from environment
    local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    if local_world_size:
        world_size = get_world_size()
        return world_size // int(local_world_size)

    # Cannot determine, assume single node
    return 1


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def cleanup():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()
