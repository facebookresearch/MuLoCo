import torch
import math
from typing import Tuple, Optional
from einops import rearrange

def bitpack_tensor(tensor, bits_per_value=2):
    """
    Pack integer values into a smaller uint8 tensor using bit packing.
    
    Args:
        tensor: Input tensor with integer values (should be uint8 or convertible)
        bits_per_value: Number of bits per value (1, 2, 4, or 8)
    
    Returns:
        torch.Tensor: Packed tensor with dtype uint8
    """
    if bits_per_value not in [1, 2, 4, 8]:
        raise ValueError("bits_per_value must be 1, 2, 4, or 8")
    
    # Convert to uint8 if needed
    if tensor.dtype != torch.uint8:
        tensor = tensor.to(torch.uint8)
    
    # Flatten the tensor to work with 1D
    original_shape = tensor.shape
    flat_tensor = tensor.flatten()
    
    # Calculate how many values we can pack per byte
    values_per_byte = 8 // bits_per_value
    
    # Pad the tensor if needed to make it divisible by values_per_byte
    remainder = len(flat_tensor) % values_per_byte
    if remainder != 0:
        padding_needed = values_per_byte - remainder
        flat_tensor = torch.cat([flat_tensor, torch.zeros(padding_needed, dtype=torch.uint8)])
    
    # Reshape to group values that will be packed together
    grouped = flat_tensor.view(-1, values_per_byte)
    
    # Pack the values using bit shifting
    packed = torch.zeros(grouped.shape[0], dtype=torch.uint8, device=grouped.device)
    
    for i in range(values_per_byte):
        shift_amount = i * bits_per_value
        packed |= (grouped[:, i] << shift_amount)
    
    return packed, original_shape

def bitunpack_tensor(packed_tensor, original_shape, bits_per_value=2):
    """
    Unpack a bit-packed tensor back to its original form.
    
    Args:
        packed_tensor: Packed tensor with dtype uint8
        original_shape: Original shape of the tensor before packing
        bits_per_value: Number of bits per value used during packing
    
    Returns:
        torch.Tensor: Unpacked tensor with original shape and dtype uint8
    """
    if bits_per_value not in [1, 2, 4, 8]:
        raise ValueError("bits_per_value must be 1, 2, 4, or 8")
    
    values_per_byte = 8 // bits_per_value
    mask = (1 << bits_per_value) - 1  # Create mask for extracting bits
    
    # Unpack values
    unpacked_values = []
    for i in range(values_per_byte):
        shift_amount = i * bits_per_value
        values = (packed_tensor >> shift_amount) & mask
        unpacked_values.append(values)
    
    # Stack values in the correct order (transpose to get row-major order)
    unpacked_matrix = torch.stack(unpacked_values, dim=1)  # Shape: [num_bytes, values_per_byte]
    flat_unpacked = unpacked_matrix.flatten()
    
    # Trim to original size and reshape
    original_size = torch.prod(torch.tensor(original_shape)).item()
    flat_unpacked = flat_unpacked[:original_size]
    
    return flat_unpacked.view(original_shape)

class QuantizationCompressor:

    def __init__(
        self,
        n_bins: int,            # Number of quantization bins (e.g., 256 for 8-bit quantization)
        range_in_sigmas: float, # Range of values to quantize, expressed in standard deviations from the mean
    ):
        """
        Initialize the TopKCompressor.
        
        Args:
            
            n_bins (int): Number of quantization levels/bins to use for statistical quantization.
                         Common values are 256 (8-bit), 16 (4-bit), 4 (2-bit), etc.
                         Must be a power of 2 for efficient bit packing.
                         Higher values = better precision but more memory usage.
            
            range_in_sigmas (float): The range of values to capture during quantization,
                                   expressed as multiples of standard deviation from the mean.
                                   For example, 3.0 means quantize values within ±3σ of the mean.
                                   Values outside this range will be clamped to the boundaries.
                                   Larger values capture more outliers but reduce precision for common values.
        """
        self.rng = None  # Random number generator (initialized later if needed)
        self.n_bins = n_bins
        
        # Calculate bits needed, but ensure it's one of the supported values (1, 2, 4, 8)
        required_bits = (n_bins - 1).bit_length()
        if required_bits <= 1:
            self.bits_per_value = 1
        elif required_bits <= 2:
            self.bits_per_value = 2
        elif required_bits <= 4:
            self.bits_per_value = 4
        elif required_bits <= 8:
            self.bits_per_value = 8
        else:
            raise ValueError(f"Unsupported number of bits: {required_bits}")
        
        self.range_in_sigmas = range_in_sigmas
        
        # Predefined dtype mapping for serialization
        self.dtype_to_code = {
            torch.float32: 0,
            torch.float64: 1,
            torch.float16: 2,
            torch.bfloat16: 3,
            torch.int32: 4,
            torch.int64: 5,
            torch.int16: 6,
            torch.int8: 7,
            torch.uint8: 8,
        }
        self.code_to_dtype = {v: k for k, v in self.dtype_to_code.items()}


    def _quantize(self, val: torch.Tensor):
        """Performs statistical 8-bit quantization."""
        offset = self.n_bins // 2
        shift = val.mean()
        centered_val = val - shift

        if centered_val.numel() <= 1:
            std_unbiased = torch.tensor(0.0, device=val.device, dtype=val.dtype)
        else:
            std_unbiased = centered_val.norm() / math.sqrt(centered_val.numel() - 1)

        scale = self.range_in_sigmas * std_unbiased / self.n_bins
        if scale == 0 or torch.isnan(scale) or torch.isinf(scale):
            scale = torch.tensor(1.0, dtype=centered_val.dtype, device=val.device)

        quantized = (
            (centered_val.float() / scale + offset)
            .round()
            .clamp(0, self.n_bins - 1)
            .to(torch.uint8)
        )

        lookup = torch.zeros(self.n_bins, dtype=torch.float32, device=val.device)
        sums = torch.zeros_like(lookup).scatter_add_(
            0, quantized.long().flatten(), centered_val.float().flatten()
        )
        counts = torch.zeros_like(lookup).scatter_add_(
            0,
            quantized.long().flatten(),
            torch.ones_like(centered_val.float().flatten()),
        )
        lookup = torch.where(counts > 0, sums / counts, 0.0)

        # Pack the quantized data
        packed, orig_shape = bitpack_tensor(quantized, bits_per_value=self.bits_per_value)
        
        # Create metadata tensor
        metadata = self._pack_metadata(shift, orig_shape, val.dtype, lookup, val.device)
        
        return packed, metadata

    def _dequantize(self, packed_tensor: torch.Tensor, metadata: torch.Tensor):
        """Dequantizes values using metadata tensor."""
        # Parse metadata tensor
        shift, orig_shape, orig_dtype, lookup = self._unpack_metadata(metadata)
        
        dequantized = bitunpack_tensor(packed_tensor, orig_shape, bits_per_value=self.bits_per_value)
        dequantized = lookup.to(packed_tensor.device)[dequantized.long()] + shift
        return dequantized.to(orig_dtype)

    def _pack_metadata(self, shift: torch.Tensor, orig_shape: tuple, orig_dtype: torch.dtype, 
                      lookup: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Pack quantization metadata into a single tensor.
        
        Args:
            shift: Mean shift value
            orig_shape: Original tensor shape
            orig_dtype: Original tensor dtype
            lookup: Lookup table for dequantization
            device: Device to create tensor on
            
        Returns:
            torch.Tensor: Packed metadata tensor with format:
                [shift, num_dims, orig_shape_dims..., dtype_code, lookup_table...]
        """
        orig_shape_tensor = torch.tensor(orig_shape, dtype=torch.int32, device=device)
        dtype_code = torch.tensor([self.dtype_to_code[orig_dtype]], dtype=torch.int32, device=device)
        
        # Concatenate all metadata into one tensor
        metadata = torch.cat([
            shift.unsqueeze(0),  # [1]
            torch.tensor([len(orig_shape)], dtype=torch.int32, device=device),  # [1] - num dims
            orig_shape_tensor,  # [num_dims]
            dtype_code,  # [1]
            lookup.flatten()  # [n_bins] - ensure it's 1D
        ])
        
        return metadata

    def _unpack_metadata(self, metadata: torch.Tensor) -> tuple:
        """
        Unpack metadata tensor back to individual components.
        
        Args:
            metadata: Packed metadata tensor
            
        Returns:
            tuple: (shift, orig_shape, orig_dtype, lookup)
        """
        # Parse metadata tensor: [shift, num_dims, orig_shape_dims..., dtype_code, lookup_table...]
        shift = metadata[0]
        num_dims = metadata[1].int().item()
        orig_shape = tuple(metadata[2:2+num_dims].int().tolist())
        dtype_code = metadata[2+num_dims].int().item()
        lookup_start = 2 + num_dims + 1
        lookup = metadata[lookup_start:lookup_start + self.n_bins]
        
        # Convert dtype code back to torch dtype using predefined mapping
        orig_dtype = self.code_to_dtype.get(dtype_code, torch.float32)
        
        return shift, orig_shape, orig_dtype, lookup



class LinearQuantizationCompressor:
    
    def __init__(
        self,
        n_bins: int,            # Number of quantization bins (e.g., 256 for 8-bit quantization)
        scale_per_tensor: bool = True,  # Whether to use per-tensor or per-channel scaling
    ):
        """
        Initialize the LinearQuantizationCompressor.
        
        Args:
            n_bins (int): Number of quantization levels/bins to use for linear quantization.
                         Common values are 256 (8-bit), 16 (4-bit), 4 (2-bit), etc.
                         Must be a power of 2 for efficient bit packing.
                         Higher values = better precision but more memory usage.
            
            scale_per_tensor (bool): Whether to use a single scale for the entire tensor (True)
                                   or per-channel scales (False). Per-channel scaling can provide
                                   better precision but requires more metadata storage.
        """
        self.rng = None  # Random number generator (initialized later if needed)
        self.n_bins = n_bins
        self.scale_per_tensor = scale_per_tensor
        
        # Calculate bits needed, but ensure it's one of the supported values (1, 2, 4, 8)
        required_bits = (n_bins - 1).bit_length()
        if required_bits <= 1:
            self.bits_per_value = 1
        elif required_bits <= 2:
            self.bits_per_value = 2
        elif required_bits <= 4:
            self.bits_per_value = 4
        elif required_bits <= 8:
            self.bits_per_value = 8
        else:
            raise ValueError(f"Unsupported number of bits: {required_bits}")
        
        # Predefined dtype mapping for serialization
        self.dtype_to_code = {
            torch.float32: 0,
            torch.float64: 1,
            torch.float16: 2,
            torch.bfloat16: 3,
            torch.int32: 4,
            torch.int64: 5,
            torch.int16: 6,
            torch.int8: 7,
            torch.uint8: 8,
        }
        self.code_to_dtype = {v: k for k, v in self.dtype_to_code.items()}

    def _quantize(self, val: torch.Tensor):
        """Performs linear quantization."""
        # Determine the range of values
        if self.scale_per_tensor:
            val_min = val.min()
            val_max = val.max()
        else:
            # Per-channel quantization (assuming last dim is channels)
            dims_to_reduce = tuple(range(val.ndim - 1))
            val_min = val.min(dim=dims_to_reduce, keepdim=True)[0]
            val_max = val.max(dim=dims_to_reduce, keepdim=True)[0]
        
        # Calculate the scale
        scale = (val_max - val_min) / (self.n_bins - 1)
        # Avoid division by zero
        scale = torch.where(scale == 0, torch.tensor(1.0, device=val.device, dtype=val.dtype), scale)
        
        # Quantize
        quantized = torch.round((val - val_min) / scale)
        # Clip to ensure values are within range
        quantized = torch.clamp(quantized, 0, self.n_bins - 1).to(torch.uint8)
        
        # Pack the quantized data
        packed, orig_shape = bitpack_tensor(quantized, bits_per_value=self.bits_per_value)
        
        # Create metadata tensor
        metadata = self._pack_metadata(val_min, val_max, scale, orig_shape, val.dtype, val.device)
        
        return packed, metadata

    def _dequantize(self, packed_tensor: torch.Tensor, metadata: torch.Tensor):
        """Dequantizes values using metadata tensor."""
        # Parse metadata tensor
        val_min, val_max, scale, orig_shape, orig_dtype = self._unpack_metadata(metadata)
        
        # Unpack the quantized data
        dequantized = bitunpack_tensor(packed_tensor, orig_shape, bits_per_value=self.bits_per_value)
        
        # Dequantize: x_dequant = x_quant * scale + x_min
        dequantized = dequantized.float() * scale + val_min
        
        return dequantized.to(orig_dtype)

    def _pack_metadata(self, val_min: torch.Tensor, val_max: torch.Tensor, scale: torch.Tensor,
                      orig_shape: tuple, orig_dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Pack quantization metadata into a single tensor.
        
        Args:
            val_min: Minimum values used for quantization
            val_max: Maximum values used for quantization  
            scale: Scale values used for quantization
            orig_shape: Original tensor shape
            orig_dtype: Original tensor dtype
            device: Device to create tensor on
            
        Returns:
            torch.Tensor: Packed metadata tensor with format:
                [num_dims, orig_shape_dims..., dtype_code, scale_per_tensor_flag, 
                 min_values..., max_values..., scale_values...]
        """
        orig_shape_tensor = torch.tensor(orig_shape, dtype=torch.int32, device=device)
        dtype_code = torch.tensor([self.dtype_to_code[orig_dtype]], dtype=torch.int32, device=device)
        scale_per_tensor_flag = torch.tensor([1 if self.scale_per_tensor else 0], dtype=torch.int32, device=device)
        
        # Flatten min, max, scale tensors to ensure they're 1D
        val_min_flat = val_min.flatten()
        val_max_flat = val_max.flatten()
        scale_flat = scale.flatten()
        
        # Concatenate all metadata into one tensor
        metadata = torch.cat([
            torch.tensor([len(orig_shape)], dtype=torch.int32, device=device),  # [1] - num dims
            orig_shape_tensor,  # [num_dims]
            dtype_code,  # [1]
            scale_per_tensor_flag,  # [1]
            torch.tensor([len(val_min_flat)], dtype=torch.int32, device=device),  # [1] - num scale elements
            val_min_flat,  # [num_scale_elements]
            val_max_flat,  # [num_scale_elements]
            scale_flat     # [num_scale_elements]
        ])
        
        return metadata

    def _unpack_metadata(self, metadata: torch.Tensor) -> tuple:
        """
        Unpack metadata tensor back to individual components.
        
        Args:
            metadata: Packed metadata tensor
            
        Returns:
            tuple: (val_min, val_max, scale, orig_shape, orig_dtype)
        """
        # Parse metadata tensor: [num_dims, orig_shape_dims..., dtype_code, scale_per_tensor_flag,
        #                        num_scale_elements, min_values..., max_values..., scale_values...]
        idx = 0
        num_dims = metadata[idx].int().item()
        idx += 1
        
        orig_shape = tuple(metadata[idx:idx+num_dims].int().tolist())
        idx += num_dims
        
        dtype_code = metadata[idx].int().item()
        idx += 1
        
        scale_per_tensor_flag = metadata[idx].int().item()
        idx += 1
        
        num_scale_elements = metadata[idx].int().item()
        idx += 1
        
        val_min_flat = metadata[idx:idx+num_scale_elements]
        idx += num_scale_elements
        
        val_max_flat = metadata[idx:idx+num_scale_elements]
        idx += num_scale_elements
        
        scale_flat = metadata[idx:idx+num_scale_elements]
        
        # Reshape min, max, scale based on scale_per_tensor flag
        if scale_per_tensor_flag == 1:
            # Per-tensor scaling - should be scalars
            val_min = val_min_flat[0]
            val_max = val_max_flat[0]
            scale = scale_flat[0]
        else:
            # Per-channel scaling - reshape to match original tensor's channel dimension
            channel_shape = (orig_shape[-1],) if len(orig_shape) > 0 else (1,)
            # Add dimensions to broadcast properly during dequantization
            broadcast_shape = [1] * len(orig_shape)
            broadcast_shape[-1] = channel_shape[0]
            
            val_min = val_min_flat.view(broadcast_shape)
            val_max = val_max_flat.view(broadcast_shape)
            scale = scale_flat.view(broadcast_shape)
        
        # Convert dtype code back to torch dtype using predefined mapping
        orig_dtype = self.code_to_dtype.get(dtype_code, torch.float32)
        
        return val_min, val_max, scale, orig_shape, orig_dtype



if __name__ == "__main__":
    import os
    import torch.distributed as dist
    from torch.distributed import init_process_group, destroy_process_group
    
    def setup_distributed():
        """Initialize distributed training."""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
        else:
            print("Not using distributed mode")
            rank = 0
            world_size = 1
            local_rank = 0
        
        if world_size > 1:
            init_process_group(backend='nccl')
        
        return rank, world_size, local_rank
    
    def test_quantized_allgather():
        """Test quantized tensor all-gather across GPUs."""
        rank, world_size, local_rank = setup_distributed()
        
        # Set device
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
        
        print(f"Rank {rank}/{world_size} on device {device}")
        tensor_size = 1024
        
        # Create test data - each rank has different data
        if world_size > 1:
            # Create different tensors for each rank
            base_tensor = torch.randn(tensor_size, tensor_size, device=device) * (rank + 1)
        else:
            base_tensor = torch.randn(tensor_size, tensor_size, device=device)
        
        print(f"Rank {rank} original tensor shape: {base_tensor.shape}, mean: {base_tensor.mean():.4f}")
        
        # Initialize quantizer
        quantizer = QuantizationCompressor(
            n_bins=4,  # 2-bit quantization
            range_in_sigmas=6.0
        )
        
        # Quantize the tensor
        packed_tensor, metadata = quantizer._quantize(base_tensor)
        print(f"Rank {rank} packed tensor shape: {packed_tensor.shape}, dtype: {packed_tensor.dtype}")
        print(f"Rank {rank} metadata shape: {metadata.shape}, dtype: {metadata.dtype}")
        
        if world_size > 1:
            # All-gather the quantized tensors
            gathered_packed = [torch.zeros_like(packed_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_packed, packed_tensor)
            
            # All-gather the metadata tensors
            gathered_metadata = [torch.zeros_like(metadata) for _ in range(world_size)]
            dist.all_gather(gathered_metadata, metadata)
            
            print(f"Rank {rank} gathered {len(gathered_packed)} packed tensors and metadata")
            
            # Dequantize each gathered tensor and collect them
            dequantized_tensors = []
            for i, (packed, meta) in enumerate(zip(gathered_packed, gathered_metadata)):
                dequantized = quantizer._dequantize(packed, meta)
                dequantized_tensors.append(dequantized)
                # print(f"Rank {rank} dequantized tensor {i} shape: {dequantized.shape}, mean: {dequantized.mean():.4f}")
                
                # Verify reconstruction quality
                if i == rank:  # Compare with original
                    mse = torch.nn.functional.mse_loss(base_tensor, dequantized)
                    print(f"Rank {rank} reconstruction MSE: {mse:.6f}")
            
            # Compute mean of all dequantized tensors (quantized all-gather approach)
            quantized_mean = torch.stack(dequantized_tensors, dim=0).mean(dim=0)
            # print(f"Rank {rank} quantized mean shape: {quantized_mean.shape}, mean: {quantized_mean.mean():.4f}")
            
            # Perform full-precision all-reduce for comparison
            full_precision_allreduce = base_tensor.clone()
            dist.all_reduce(full_precision_allreduce, op=dist.ReduceOp.SUM)
            full_precision_allreduce = full_precision_allreduce / world_size
            
            # print(f"Rank {rank} full-precision all-reduce shape: {full_precision_allreduce.shape}, mean: {full_precision_allreduce.mean():.4f}")
            
            # Compare the two approaches
            comparison_mse = torch.nn.functional.mse_loss(quantized_mean, full_precision_allreduce)
            import time 
            time.sleep(0.3 * rank)
            print(f"Rank {rank} MSE between quantized mean and full-precision all-reduce: {comparison_mse:.8f}")
            
            # Additional statistics
            max_diff = torch.abs(quantized_mean - full_precision_allreduce).max()
            mean_abs_diff = torch.abs(quantized_mean - full_precision_allreduce).mean()
            # print(f"Rank {rank} Max absolute difference: {max_diff:.8f}")
            # print(f"Rank {rank} Mean absolute difference: {mean_abs_diff:.8f}")
            
            # Run ring all-reduce simulation on rank 0 only
            if rank == 0:
                # Gather all original tensors for simulation
                all_original_tensors = []
                for i in range(world_size):
                    # Each rank's original tensor (we have them from gathered_packed)
                    if i == rank:
                        all_original_tensors.append(base_tensor)
                    else:
                        # For other ranks, we need to reconstruct from their quantized versions
                        # Since we don't have access to other ranks' original tensors in this context,
                        # we'll use the dequantized versions as approximations
                        all_original_tensors.append(dequantized_tensors[i])
                
                # Run the ring all-reduce simulation
                ring_result, ring_baseline, ring_mse = simulate_ring_allreduce_with_quantization(
                    all_original_tensors, quantizer, rank
                )
                
                print(f"\n=== Ring vs All-Gather Comparison ===")
                print(f"All-gather mean MSE: {comparison_mse:.8f}")
                print(f"Ring all-reduce MSE: {ring_mse:.8f}")
                
                # Compare ring result with all-gather result
                ring_vs_allgather_mse = torch.nn.functional.mse_loss(quantized_mean, ring_result)
                print(f"Ring vs All-gather MSE: {ring_vs_allgather_mse:.8f}")

        else:
            # Single GPU test - just dequantize
            dequantized = quantizer._dequantize(packed_tensor, metadata)
            mse = torch.nn.functional.mse_loss(base_tensor, dequantized)
            print(f"Single GPU reconstruction MSE: {mse:.6f}")
        
        if world_size > 1:
            destroy_process_group()
        
            print(f"Rank {rank} test completed successfully!")
    
    def simulate_ring_allreduce_with_quantization(original_tensors, quantizer, rank=0):
        """
        Simulate ring-all-reduce with quantization on GPU0 only.
        Each rank communicates only a slice (shard) of the tensor at each step.
        
        Args:
            original_tensors: List of tensors from all ranks (should be gathered on rank 0)
            quantizer: QuantizationCompressor instance
            rank: Current rank (should be 0 for this simulation)
        
        Returns:
            tuple: (ring_result, full_precision_baseline, mse)
        """
        world_size = len(original_tensors)
        device = original_tensors[0].device
        tensor_shape = original_tensors[0].shape
        
        print(f"\n=== Ring All-Reduce Simulation (World Size: {world_size}) ===")
        print(f"Tensor shape: {tensor_shape}")
        
        # Determine shard size - for simplicity, we'll shard along the first dimension
        # In practice, this could be any dimension or even a custom sharding strategy
        total_elements = tensor_shape[0]
        shard_size = total_elements // world_size
        if total_elements % world_size != 0:
            print(f"Warning: Tensor size {total_elements} not evenly divisible by world_size {world_size}")
            print(f"Using shard_size = {shard_size}, last rank will handle remainder")
        
        print(f"Shard size per rank: {shard_size}")
        
        # Step 1: Determine the path for each rank's shard
        print("\nRing All-Reduce Path Visualization:")
        print("=" * 80)
        
        # Create a matrix to track how each rank's shard flows through the ring
        shard_paths = {}
        for src_rank in range(world_size):
            path = []
            current_rank = src_rank
            steps = 0
            
            print(f"\nRank {src_rank} shard path (shard {src_rank}):")
            while steps < world_size:
                if steps == 0:
                    print(f"  Step {steps}: Start at rank {current_rank} (original shard {src_rank})")
                    path.append((current_rank, "original", "quantize_shard"))
                else:
                    next_rank = (current_rank + 1) % world_size
                    print(f"  Step {steps}: Move shard {src_rank} from rank {current_rank} to rank {next_rank}")
                    path.append((next_rank, "received", "dequantize_add_requantize_shard"))
                    current_rank = next_rank
                steps += 1
            
            shard_paths[src_rank] = path
        
        print("\n" + "=" * 80)
        
        # Step 2: Simulate the ring all-reduce
        print("\nSimulating Ring All-Reduce Operations:")
        print("-" * 50)
        
        # Initialize working tensors for each rank (each rank starts with its own tensor)
        rank_tensors = {}
        for r in range(world_size):
            rank_tensors[r] = original_tensors[r].clone()
            print(f"Rank {r} initial tensor shape: {rank_tensors[r].shape}, mean: {rank_tensors[r].mean():.6f}")
        
        # Initialize accumulated shard storage for each rank
        # This tracks the accumulated sum for each shard at each rank
        rank_accumulated_shards = {}
        for r in range(world_size):
            rank_accumulated_shards[r] = {}  # Will store accumulated shards from different source ranks
        
        # Simulate each step of the ring all-reduce
        for step in range(world_size):
            print(f"\n--- Ring Step {step} ---")
            
            if step == 0:
                # First step: each rank quantizes and sends its own shard
                for src_rank in range(world_size):
                    # Extract the shard that this rank is responsible for
                    start_idx = src_rank * shard_size
                    end_idx = start_idx + shard_size if src_rank < world_size - 1 else tensor_shape[0]
                    
                    # Get the shard from the original tensor
                    shard_to_send = original_tensors[src_rank][start_idx:end_idx].clone()
                    
                    # Quantize the shard
                    packed, metadata = quantizer._quantize(shard_to_send)
                    dequantized_shard = quantizer._dequantize(packed, metadata)
                    
                    print(f"  Rank {src_rank}: Quantize shard {src_rank} (indices {start_idx}:{end_idx})")
                    print(f"    Shard shape: {shard_to_send.shape}, quantization MSE: {torch.nn.functional.mse_loss(shard_to_send, dequantized_shard):.8f}")
                    
                    # Initialize the accumulated shard at the source rank
                    rank_accumulated_shards[src_rank][src_rank] = dequantized_shard.clone()
                    
                    # Send to next rank
                    next_rank = (src_rank + 1) % world_size
                    if next_rank not in rank_accumulated_shards:
                        rank_accumulated_shards[next_rank] = {}
                    rank_accumulated_shards[next_rank][src_rank] = dequantized_shard
                    
            else:
                # Subsequent steps: receive shard, add local portion, and send to next rank
                for src_rank in range(world_size):
                    current_rank = (src_rank + step) % world_size
                    next_rank = (current_rank + 1) % world_size
                    
                    if current_rank in rank_accumulated_shards and src_rank in rank_accumulated_shards[current_rank]:
                        # Get the accumulated shard that was received
                        received_accumulated_shard = rank_accumulated_shards[current_rank][src_rank]
                        
                        # Extract the local unquantized portion for this shard
                        start_idx = src_rank * shard_size
                        end_idx = start_idx + shard_size if src_rank < world_size - 1 else tensor_shape[0]
                        local_shard_portion = original_tensors[current_rank][start_idx:end_idx].clone()
                        
                        # Add the local unquantized portion to the received accumulated shard
                        new_accumulated_shard = received_accumulated_shard + local_shard_portion
                        
                        print(f"  Rank {current_rank}: Add local portion to shard {src_rank}")
                        print(f"    Received accumulated mean: {received_accumulated_shard.mean():.6f}")
                        print(f"    Local portion mean: {local_shard_portion.mean():.6f}")
                        print(f"    New accumulated mean: {new_accumulated_shard.mean():.6f}")
                        
                        # Update the accumulated shard at current rank
                        rank_accumulated_shards[current_rank][src_rank] = new_accumulated_shard.clone()
                        
                        if step < world_size - 1:  # Not the last step
                            # Quantize the new accumulated shard for transmission
                            packed, metadata = quantizer._quantize(new_accumulated_shard)
                            dequantized_accumulated_shard = quantizer._dequantize(packed, metadata)
                            
                            print(f"    Requantization MSE: {torch.nn.functional.mse_loss(new_accumulated_shard, dequantized_accumulated_shard):.8f}")
                            
                            # Send to next rank
                            if next_rank not in rank_accumulated_shards:
                                rank_accumulated_shards[next_rank] = {}
                            rank_accumulated_shards[next_rank][src_rank] = dequantized_accumulated_shard
                        else:
                            print(f"    Final step - no transmission needed")
        
        # Reconstruct the final result by combining all accumulated shards
        # Each rank should have the fully accumulated version of its responsible shard
        final_result = torch.zeros_like(original_tensors[0])
        
        for src_rank in range(world_size):
            # The final accumulated shard for src_rank should be at rank src_rank
            start_idx = src_rank * shard_size
            end_idx = start_idx + shard_size if src_rank < world_size - 1 else tensor_shape[0]
            
            if src_rank in rank_accumulated_shards and src_rank in rank_accumulated_shards[src_rank]:
                final_result[start_idx:end_idx] = rank_accumulated_shards[src_rank][src_rank]
                print(f"Final shard {src_rank} mean: {rank_accumulated_shards[src_rank][src_rank].mean():.6f}")
            else:
                print(f"Warning: Missing final accumulated shard {src_rank}")
        
        ring_result = final_result
        
        # The final result should be the sum of all original tensors
        ring_result = rank_tensors[0].clone()
        
        # Compute full-precision baseline
        full_precision_baseline = torch.stack(original_tensors, dim=0).sum(dim=0)
        
        # Calculate MSE
        mse = torch.nn.functional.mse_loss(ring_result, full_precision_baseline)
        
        print(f"\n=== Results ===")
        print(f"Ring all-reduce result shape: {ring_result.shape}")
        print(f"Ring all-reduce result mean: {ring_result.mean():.6f}")
        print(f"Full-precision baseline mean: {full_precision_baseline.mean():.6f}")
        print(f"MSE between ring result and full-precision baseline: {mse:.8f}")
        
        # Additional statistics
        max_diff = torch.abs(ring_result - full_precision_baseline).max()
        mean_abs_diff = torch.abs(ring_result - full_precision_baseline).mean()
        print(f"Max absolute difference: {max_diff:.8f}")
        print(f"Mean absolute difference: {mean_abs_diff:.8f}")
        
        return ring_result, full_precision_baseline, mse
    
    # Run the test
    test_quantized_allgather()



