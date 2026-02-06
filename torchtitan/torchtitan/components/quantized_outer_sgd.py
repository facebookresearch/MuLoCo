"""
Quantized Outer SGD: Low-Communication Optimizers for Distributed Training.

This module provides implementations of communication-efficient optimizers for
distributed deep learning, combining gradient compression techniques with local
optimization strategies to reduce bandwidth requirements during training.

Code adapted from:
    - SparseLoCo: https://github.com/one-covenant/SparseLoCo
      (Communication Efficient LLM Pre-training with SparseLoCo)
    - DeMo: https://github.com/bloc97/DeMo
      (Decoupled Momentum Optimization)

================================================================================
HELPER FUNCTIONS
================================================================================

bitpack_tensor(tensor, bits_per_value=2)
    Pack integer values into a smaller uint8 tensor using bit packing.
    Supports 1, 2, 4, or 8 bits per value for efficient storage.

bitunpack_tensor(packed_tensor, original_shape, bits_per_value=2)
    Unpack a bit-packed tensor back to its original form.
    Reverses the bitpack_tensor operation.

================================================================================
COMPRESSOR CLASSES
================================================================================

QuantizationCompressor
    Statistical quantization compressor using per-tensor statistics.
    Quantizes values based on mean and standard deviation, with a configurable
    number of bins and sigma range. Uses lookup tables for accurate reconstruction.

BatchedRowiseQuantizationCompressor
    Vectorized row-wise statistical quantization compressor.
    Performs per-row quantization using batched operations for efficiency.
    Each row has its own statistics and lookup table.

LinearQuantizationCompressor
    Linear min-max quantization compressor.
    Quantizes values linearly between min and max values.
    Supports per-tensor or per-channel scaling.

RowWiseLinearQuantizationCompressor
    Row-wise linear quantization compressor.
    Performs per-row linear quantization with independent min/max/scale per row.
    Uses fully batched operations for efficiency.

TopKCompressor
    Top-K sparsification compressor.
    Keeps only the largest magnitude values (by a configurable sparsity ratio)
    and discards the rest. Useful for sparse gradient communication.

================================================================================
OPTIMIZER CLASSES
================================================================================

QuantizedOuterSGD(torch.optim.SGD)
    Quantized outer optimizer with error feedback for DiLoCo-style training.
    
    Features:
    - Multiple compression strategies (statistical, linear, row-wise, top-k)
    - Error feedback mechanism to accumulate compression residuals
    - Support for skipping quantization on embeddings and layer norms
    - Simulated quantization after reduce for bandwidth estimation
    
    Methods:
    - _save_grads(): Compress and save pseudogradients for communication
    - _set_grads(): Decompress gathered gradients and set parameter gradients
    - step(): Perform the outer optimizer update

WorkerGroup
    Helper class for managing worker groups in hierarchical communication.
    Handles both flat worker lists and nested group structures for TreeLoCo.
    
    Methods:
    - get_indices_to_avg(): Get representative indices for averaging
    - get_all_workers(): Flatten and return all worker indices

================================================================================
"""

# Standard library
import math
import logging
import os
from typing import (
    Literal,
    Optional,
    Callable,
    List,
    Tuple,
    Union,
    TypeAlias,
    Iterable,
    Any,
)

# Third party
import torch
import torch.fft
import torch.distributed as dist
from einops import rearrange

# ---------- Type Aliases ---------- #
ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[dict[str, Any]]]

# --------- Helper Classes --------- #
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
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
    
    # Get the device from the input tensor
    device = tensor.device
    
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
        flat_tensor = torch.cat([flat_tensor, torch.zeros(padding_needed, dtype=torch.uint8, device=device)])
    
    # Reshape to group values that will be packed together
    grouped = flat_tensor.view(-1, values_per_byte)
    
    # Pack the values using bit shifting
    packed = torch.zeros(grouped.shape[0], dtype=torch.uint8, device=device)
    
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
    
    # Get the device from the input tensor
    device = packed_tensor.device
    
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
    original_size = torch.prod(torch.tensor(original_shape, device=device)).item()
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
        """Performs statistical quantization using efficient vectorized operations."""
        orig_shape = val.shape
        offset = self.n_bins // 2
        
        # Compute tensor-level statistics - use more efficient operations
        shift = val.mean()
        centered_val = val - shift

        if centered_val.numel() <= 1:
            std_unbiased = torch.tensor(0.0, device=val.device, dtype=val.dtype)
        else:
            # Use var() which is more efficient than norm() for std calculation
            std_unbiased = torch.sqrt(centered_val.var(unbiased=True))

        scale = self.range_in_sigmas * std_unbiased / self.n_bins
        if scale == 0 or torch.isnan(scale) or torch.isinf(scale):
            scale = torch.tensor(1.0, dtype=centered_val.dtype, device=val.device)

        # Quantize using vectorized operations - avoid unnecessary type conversions
        quantized = torch.clamp(
            torch.round(centered_val / scale + offset),
            0, self.n_bins - 1
        ).to(torch.uint8)

        # Create lookup table using more efficient bincount approach
        flat_quantized = quantized.flatten()
        flat_centered = centered_val.flatten()
        
        # Use bincount for much faster histogram computation
        # This is significantly faster than scatter_add for this use case
        bin_counts = torch.bincount(flat_quantized.long(), minlength=self.n_bins).float()
        bin_sums = torch.bincount(flat_quantized.long(), weights=flat_centered.float(), minlength=self.n_bins)
        
        # Compute lookup table with safe division - avoid where() overhead
        lookup = bin_sums / torch.clamp(bin_counts, min=1e-8)

        # Pack the quantized data
        packed, _ = bitpack_tensor(quantized, bits_per_value=self.bits_per_value)
        
        # Create metadata tensor
        metadata = self._pack_metadata(shift, orig_shape, val.dtype, lookup, val.device)
        
        return packed, metadata

    def _dequantize(self, packed_tensor: torch.Tensor, metadata: torch.Tensor):
        """Dequantizes values using metadata tensor with vectorized operations."""
        # Parse metadata tensor
        shift, orig_shape, orig_dtype, lookup = self._unpack_metadata(metadata)
        
        # Unpack quantized values
        quantized = bitunpack_tensor(packed_tensor, orig_shape, bits_per_value=self.bits_per_value)
        
        # Vectorized lookup using advanced indexing - ensure indices are long
        # Move lookup to same device as packed_tensor to avoid device transfers
        lookup_device = lookup.to(packed_tensor.device)
        dequantized = lookup_device[quantized.long()] + shift
        
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
        # Ensure all tensors are on the target device before concatenation
        shift_device = shift.to(device)
        lookup_device = lookup.to(device)
        
        # Create tensors directly on target device to avoid transfers
        orig_shape_tensor = torch.tensor(orig_shape, dtype=torch.float32, device=device)
        dtype_code = torch.tensor([self.dtype_to_code[orig_dtype]], dtype=torch.float32, device=device)
        num_dims_tensor = torch.tensor([len(orig_shape)], dtype=torch.float32, device=device)
        
        # Concatenate all metadata into one tensor - use consistent dtype
        metadata = torch.cat([
            shift_device.unsqueeze(0).float(),  # [1]
            num_dims_tensor,  # [1] - num dims
            orig_shape_tensor,  # [num_dims]
            dtype_code,  # [1]
            lookup_device.flatten()  # [n_bins] - ensure it's 1D
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
        num_dims = int(metadata[1].item())
        orig_shape = tuple(int(x.item()) for x in metadata[2:2+num_dims])
        dtype_code = int(metadata[2+num_dims].item())
        lookup_start = 2 + num_dims + 1
        lookup = metadata[lookup_start:lookup_start + self.n_bins]
        
        # Convert dtype code back to torch dtype using predefined mapping
        orig_dtype = self.code_to_dtype.get(dtype_code, torch.float32)
        
        return shift, orig_shape, orig_dtype, lookup

class BatchedRowiseQuantizationCompressor:
    """
    A vectorized version of RowiseQuantizationCompressor that avoids for loops
    by using batched operations and PyTorch's built-in functions.
    """

    def __init__(
        self,
        n_bins: int,            # Number of quantization bins (e.g., 256 for 8-bit quantization)
        range_in_sigmas: float, # Range of values to quantize, expressed in standard deviations from the mean
    ):
        """
        Initialize the BatchedRowiseQuantizationCompressor.
        
        Args:
            n_bins (int): Number of quantization levels/bins to use for statistical quantization.
            range_in_sigmas (float): The range of values to capture during quantization,
                                   expressed as multiples of standard deviation from the mean.
        """
        self.n_bins = n_bins
        self.range_in_sigmas = range_in_sigmas
        
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
        """Performs row-wise statistical quantization using vectorized operations."""
        orig_shape = val.shape
        
        # Handle 1D tensors by treating them as a single row
        if val.dim() == 1:
            val_2d = val.unsqueeze(0)  # Shape: [1, N]
        else:
            # Flatten to 2D: [num_rows, elements_per_row]
            val_2d = val.view(val.shape[0], -1)
        
        num_rows, elements_per_row = val_2d.shape
        offset = self.n_bins // 2
        
        # Compute row-wise statistics (vectorized)
        shifts = val_2d.mean(dim=1, keepdim=True)  # Shape: [num_rows, 1]
        centered_val = val_2d - shifts
        
        # Compute row-wise standard deviations (vectorized)
        # For unbiased std: std = ||x - mean|| / sqrt(n-1)
        row_norms = centered_val.norm(dim=1, keepdim=True)  # [num_rows, 1]
        # Handle case where elements_per_row <= 1
        sqrt_n_minus_1 = math.sqrt(max(elements_per_row - 1, 1))
        std_unbiased = row_norms / sqrt_n_minus_1
        
        scales = self.range_in_sigmas * std_unbiased / self.n_bins
        # Handle zero/invalid scales
        scales = torch.where(
            (scales == 0) | torch.isnan(scales) | torch.isinf(scales),
            torch.ones_like(scales),
            scales
        )
        
        # Quantize (vectorized)
        quantized = (
            (centered_val.float() / scales + offset)
            .round()
            .clamp(0, self.n_bins - 1)
            .to(torch.uint8)
        )
        
        # Create lookup tables using scatter operations (vectorized)
        # Expand quantized indices to include batch dimension
        batch_indices = torch.arange(num_rows, device=val.device).unsqueeze(1).expand(-1, elements_per_row)
        flat_batch_indices = batch_indices.flatten()
        flat_quantized = quantized.flatten()
        flat_centered = centered_val.float().flatten()
        
        # Create combined indices for scatter operations
        combined_indices = flat_batch_indices * self.n_bins + flat_quantized.long()
        
        # Compute sums and counts for each (row, bin) combination
        total_bins = num_rows * self.n_bins
        sums = torch.zeros(total_bins, dtype=torch.float32, device=val.device)
        counts = torch.zeros(total_bins, dtype=torch.float32, device=val.device)
        
        sums.scatter_add_(0, combined_indices, flat_centered)
        counts.scatter_add_(0, combined_indices, torch.ones_like(flat_centered))
        
        # Reshape and compute lookup tables
        sums = sums.view(num_rows, self.n_bins)
        counts = counts.view(num_rows, self.n_bins)
        lookups = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))
        
        # Reshape quantized back to original shape
        if val.dim() == 1:
            quantized = quantized.squeeze(0)
        else:
            quantized = quantized.view(orig_shape)


        packed, _ = bitpack_tensor(quantized, bits_per_value=self.bits_per_value)
        
        # Create metadata tensor (no packing for now)
        metadata = self._pack_metadata(shifts.squeeze(1), orig_shape, val.dtype, lookups, val.device)
        
        return packed, metadata

    def _dequantize(self, quantized_tensor: torch.Tensor, metadata: torch.Tensor):
        """Dequantizes values using row-wise metadata tensor (vectorized)."""
        # Parse metadata tensor
        shifts, orig_shape, orig_dtype, lookups = self._unpack_metadata(metadata)
        
        quantized_tensor = bitunpack_tensor(quantized_tensor, orig_shape, bits_per_value=self.bits_per_value)
        
        # Handle 1D tensors
        if len(orig_shape) == 1:
            quantized_2d = quantized_tensor.unsqueeze(0)
        else:
            quantized_2d = quantized_tensor.view(orig_shape[0], -1)
        
        num_rows, elements_per_row = quantized_2d.shape
        
        # Vectorized lookup using advanced indexing
        # Create row indices for each element
        row_indices = torch.arange(num_rows, device=quantized_tensor.device).unsqueeze(1).expand(-1, elements_per_row)
        
        # Use advanced indexing to perform lookup
        dequantized_values = lookups[row_indices, quantized_2d.long()]
        
        # Add back the shifts (vectorized)
        result = dequantized_values + shifts.unsqueeze(1)
        
        # Reshape back to original shape
        if len(orig_shape) == 1:
            result = result.squeeze(0)
        else:
            result = result.view(orig_shape)
        
        return result.to(orig_dtype)

    def _pack_metadata(self, shifts: torch.Tensor, orig_shape: tuple, orig_dtype: torch.dtype, 
                      lookups: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Pack row-wise quantization metadata into a single tensor."""
        orig_shape_tensor = torch.tensor(orig_shape, dtype=torch.int32, device=device)
        dtype_code = torch.tensor([self.dtype_to_code[orig_dtype]], dtype=torch.int32, device=device)
        num_rows = torch.tensor([shifts.shape[0]], dtype=torch.int32, device=device)
        
        # Concatenate all metadata into one tensor
        metadata = torch.cat([
            torch.tensor([len(orig_shape)], dtype=torch.int32, device=device),  # [1] - num dims
            orig_shape_tensor,  # [num_dims]
            dtype_code,  # [1]
            num_rows,  # [1]
            shifts.flatten(),  # [num_rows]
            lookups.flatten()  # [num_rows * n_bins]
        ])
        
        return metadata

    def _unpack_metadata(self, metadata: torch.Tensor) -> tuple:
        """Unpack row-wise metadata tensor back to individual components."""
        # Parse metadata tensor: [num_dims, orig_shape_dims..., dtype_code, num_rows, shifts..., lookups_flattened...]
        idx = 0
        num_dims = metadata[idx].int().item()
        idx += 1
        
        orig_shape = tuple(metadata[idx:idx+num_dims].int().tolist())
        idx += num_dims
        
        dtype_code = metadata[idx].int().item()
        idx += 1
        
        num_rows = metadata[idx].int().item()
        idx += 1
        
        shifts = metadata[idx:idx+num_rows]
        idx += num_rows
        
        lookups_flat = metadata[idx:idx + num_rows * self.n_bins]
        lookups = lookups_flat.view(num_rows, self.n_bins)
        
        # Convert dtype code back to torch dtype using predefined mapping
        orig_dtype = self.code_to_dtype.get(dtype_code, torch.float32)
        
        return shifts, orig_shape, orig_dtype, lookups


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


class RowWiseLinearQuantizationCompressor:
    
    def __init__(
        self,
        n_bins: int,            # Number of quantization bins (e.g., 256 for 8-bit quantization)
    ):
        """
        Initialize the RowWiseLinearQuantizationCompressor.
        
        Args:
            n_bins (int): Number of quantization levels/bins to use for linear quantization.
                         Common values are 256 (8-bit), 16 (4-bit), 4 (2-bit), etc.
                         Must be a power of 2 for efficient bit packing.
                         Higher values = better precision but more memory usage.
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
        """Performs row-wise linear quantization using fully batched operations."""
        # Store original shape and dtype
        original_shape = val.shape
        original_dtype = val.dtype
        device = val.device
        
        # Flatten tensor to 2D: [num_rows, elements_per_row]
        if val.ndim == 1:
            val_2d = val.unsqueeze(0)
            num_rows = 1
        else:
            num_rows = val.shape[0]
            val_2d = val.view(num_rows, -1)
        
        elements_per_row = val_2d.shape[1]
        
        # Batched min/max computation across rows
        row_mins, _ = val_2d.min(dim=1, keepdim=True)  # [num_rows, 1]
        row_maxs, _ = val_2d.max(dim=1, keepdim=True)  # [num_rows, 1]
        
        # Batched scale computation
        ranges = row_maxs - row_mins
        scales = ranges / (self.n_bins - 1)
        # Avoid division by zero for constant rows
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        
        # Batched quantization - fully vectorized
        normalized = (val_2d - row_mins) / scales  # [num_rows, elements_per_row]
        quantized_vals = torch.round(normalized)
        quantized_vals = torch.clamp(quantized_vals, 0, self.n_bins - 1)
        
        # Convert to uint8 for packing
        quantized_uint8 = quantized_vals.to(torch.uint8)
        
        # Reshape back to original shape for bit packing
        if val.ndim == 1:
            quantized_reshaped = quantized_uint8.squeeze(0)
        else:
            quantized_reshaped = quantized_uint8.view(original_shape)
        
        # Use bitpack_tensor for efficient bit packing
        packed_tensor, _ = bitpack_tensor(quantized_reshaped, bits_per_value=self.bits_per_value)
        
        # Create metadata tensor
        metadata = self._pack_metadata(row_mins.squeeze(1), row_maxs.squeeze(1), 
                                     scales.squeeze(1), original_shape, original_dtype, device)
        
        return packed_tensor, metadata

    def _dequantize(self, packed_tensor: torch.Tensor, metadata: torch.Tensor):
        """Dequantizes values using fully batched operations."""
        # Unpack metadata
        row_mins, row_maxs, scales, original_shape, orig_dtype = self._unpack_metadata(metadata)
        
        # Use bitunpack_tensor for efficient bit unpacking
        quantized_tensor = bitunpack_tensor(packed_tensor, original_shape, bits_per_value=self.bits_per_value)
        
        # Flatten tensor to 2D for processing
        if len(original_shape) == 1:
            quantized_2d = quantized_tensor.unsqueeze(0)
        else:
            quantized_2d = quantized_tensor.view(original_shape[0], -1)
        
        # Batched dequantization - fully vectorized
        row_mins_expanded = row_mins.unsqueeze(1)  # [num_rows, 1]
        scales_expanded = scales.unsqueeze(1)      # [num_rows, 1]
        
        dequantized_tensor = quantized_2d.float() * scales_expanded + row_mins_expanded
        
        # Reshape to original shape
        if len(original_shape) == 1:
            result = dequantized_tensor.squeeze(0)
        else:
            result = dequantized_tensor.view(original_shape)
        
        return result.to(orig_dtype)

    def _pack_metadata(self, row_mins: torch.Tensor, row_maxs: torch.Tensor, 
                      scales: torch.Tensor, orig_shape: tuple, orig_dtype: torch.dtype, 
                      device: torch.device) -> torch.Tensor:
        """Pack row-wise linear quantization metadata into a single tensor."""
        orig_shape_tensor = torch.tensor(orig_shape, dtype=torch.int32, device=device)
        dtype_code = torch.tensor([self.dtype_to_code[orig_dtype]], dtype=torch.int32, device=device)
        num_rows = torch.tensor([row_mins.shape[0]], dtype=torch.int32, device=device)
        
        # Concatenate all metadata into one tensor
        metadata = torch.cat([
            torch.tensor([len(orig_shape)], dtype=torch.int32, device=device),  # [1] - num dims
            orig_shape_tensor,  # [num_dims]
            dtype_code,  # [1]
            num_rows,  # [1]
            row_mins.flatten(),  # [num_rows]
            row_maxs.flatten(),  # [num_rows]
            scales.flatten()     # [num_rows]
        ])
        
        return metadata

    def _unpack_metadata(self, metadata: torch.Tensor) -> tuple:
        """Unpack row-wise linear quantization metadata tensor back to individual components."""
        # Parse metadata tensor: [num_dims, orig_shape_dims..., dtype_code, num_rows, mins..., maxs..., scales...]
        idx = 0
        num_dims = metadata[idx].int().item()
        idx += 1
        
        orig_shape = tuple(metadata[idx:idx+num_dims].int().tolist())
        idx += num_dims
        
        dtype_code = metadata[idx].int().item()
        idx += 1
        
        num_rows = metadata[idx].int().item()
        idx += 1
        
        row_mins = metadata[idx:idx+num_rows]
        idx += num_rows
        
        row_maxs = metadata[idx:idx+num_rows]
        idx += num_rows
        
        scales = metadata[idx:idx+num_rows]
        
        # Convert dtype code back to torch dtype using predefined mapping
        orig_dtype = self.code_to_dtype.get(dtype_code, torch.float32)
        
        return row_mins, row_maxs, scales, orig_shape, orig_dtype





class TopKCompressor:
    
    def __init__(
        self,
        sparsity_ratio: float = 0.1,  # Percentage of entries to keep (0.1 = 10%)
    ):
        """
        Initialize the TopKCompressor.
        
        Args:
            sparsity_ratio (float): Fraction of entries to keep (between 0 and 1).
                                   0.1 means keep top 10% of values by magnitude.
        """
        if not 0 < sparsity_ratio <= 1:
            raise ValueError("sparsity_ratio must be between 0 and 1")
        
        self.sparsity_ratio = sparsity_ratio
        
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
        """Performs TopK compression by keeping only the largest magnitude values."""
        # Store original properties
        original_shape = val.shape
        original_dtype = val.dtype
        device = val.device
        
        # Flatten tensor for processing
        flat_val = val.flatten()
        total_elements = flat_val.numel()
        
        # Calculate number of elements to keep
        k = max(1, int(total_elements * self.sparsity_ratio))
        
        # Find TopK indices by magnitude
        abs_vals = torch.abs(flat_val)
        _, topk_indices = torch.topk(abs_vals, k, largest=True, sorted=False)
        
        # Extract TopK values
        topk_values = flat_val[topk_indices]
        
        # Create metadata tensor
        metadata = self._pack_metadata(topk_indices, original_shape, original_dtype, device)
        
        return topk_values, metadata

    def _dequantize(self, topk_values: torch.Tensor, metadata: torch.Tensor):
        """Reconstructs the sparse tensor from TopK values and indices."""
        # Unpack metadata
        topk_indices, original_shape, orig_dtype = self._unpack_metadata(metadata)
        
        # Create sparse tensor
        total_elements = torch.prod(torch.tensor(original_shape)).item()
        reconstructed_flat = torch.zeros(total_elements, dtype=topk_values.dtype, device=topk_values.device)
        
        # Place TopK values at their original positions
        reconstructed_flat[topk_indices] = topk_values
        
        # Reshape to original shape and convert to original dtype
        result = reconstructed_flat.view(original_shape).to(orig_dtype)
        
        return result

    def _pack_metadata(self, topk_indices: torch.Tensor, orig_shape: tuple, 
                      orig_dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Pack TopK compression metadata into a single tensor."""
        orig_shape_tensor = torch.tensor(orig_shape, dtype=torch.int64, device=device)
        dtype_code = torch.tensor([self.dtype_to_code[orig_dtype]], dtype=torch.int64, device=device)
        k = torch.tensor([topk_indices.numel()], dtype=torch.int64, device=device)
        
        # Concatenate all metadata into one tensor
        metadata = torch.cat([
            torch.tensor([len(orig_shape)], dtype=torch.int64, device=device),  # [1] - num dims
            orig_shape_tensor,  # [num_dims]
            dtype_code,  # [1]
            k,  # [1] - number of TopK elements
            topk_indices.long()  # [k] - indices of TopK elements
        ])
        
        return metadata

    def _unpack_metadata(self, metadata: torch.Tensor) -> tuple:
        """Unpack TopK compression metadata tensor back to individual components."""
        # Parse metadata tensor: [num_dims, orig_shape_dims..., dtype_code, k, indices...]
        idx = 0
        num_dims = metadata[idx].int().item()
        idx += 1
        
        orig_shape = tuple(metadata[idx:idx+num_dims].int().tolist())
        idx += num_dims
        
        dtype_code = metadata[idx].int().item()
        idx += 1
        
        k = metadata[idx].int().item()
        idx += 1
        
        topk_indices = metadata[idx:idx+k]
        
        # Convert dtype code back to torch dtype
        orig_dtype = self.code_to_dtype.get(dtype_code, torch.float32)
        
        return topk_indices, orig_shape, orig_dtype


class QuantizedOuterSGD(torch.optim.SGD):
    """Implements the Quantized Outer SGD optimizer."""

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        maximize: bool = False,
        error_decay: float = 0.95,
        error_feedback_alpha: float = 1.0,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        quantization_bins: int = 256,
        quantization_range: int = 6,
        compressor_type: str = "statistical",
        topk_sparsity_ratio: float = 0.1,
        skip_norm_quantization: bool = False,
        skip_embedding_quantization: bool = False,
        simulate_quantization_after_reduce: bool = True,
        use_ef: bool = True,
    ):
        super().__init__(params, lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=0.0, maximize=maximize)# **kwargs)

        self.error_decay = error_decay
        self.error_feedback_alpha = error_feedback_alpha
        self.lr = lr
        self.skip_norm_quantization = skip_norm_quantization
        self.skip_embedding_quantization = skip_embedding_quantization
        self.use_ef = use_ef
        self.compressor_type = compressor_type

        if compressor_type == "topk" and simulate_quantization_after_reduce:
            print("Simulate quantization after reduce is not supported for topk compressor, setting to False")
            self.simulate_quantization_after_reduce = False
        else:
            self.simulate_quantization_after_reduce = simulate_quantization_after_reduce


        if compressor_type == "statistical":
            self.compressor = QuantizationCompressor(
            quantization_bins,
            quantization_range,
        )
        elif compressor_type == "row_wise_statistical":
            self.compressor = BatchedRowiseQuantizationCompressor(
                quantization_bins,
                quantization_range,
            )
        elif compressor_type == "linear":
            self.compressor = LinearQuantizationCompressor(
                quantization_bins,
            )
        elif compressor_type == "row_wise_linear":
            self.compressor = RowWiseLinearQuantizationCompressor(
                quantization_bins,
            )
        elif compressor_type == "topk":
            self.compressor = TopKCompressor(
                topk_sparsity_ratio,
            )
        else:
            raise ValueError(f"Invalid compressor type: {compressor_type}")


        if self.use_ef:
            for group in self.param_groups:
                #TODO skip EF creations when skipping embeddings or norms
                for p in group["params"]:
                    if p.requires_grad:
                        self.state[p]["error_buffer"] = torch.zeros_like(p)



    def _is_param_embedding(self, param_name: str) -> bool:
        if "embedding" in param_name or "output.weight" in param_name:
            return True
        return False
    
    def _save_grads(
            self, 
            grads: Dict[str, torch.Tensor], 
            previous_params: Dict[str, torch.Tensor], 
            fragment: Any
        ):
        """
        Saves compressed gradients to be communicated across workers.
        """
        with torch.no_grad():
            for param_name, p in fragment.named_parameters():
                if isinstance(p, DTensor):
                    local_param = p.to_local()
                else:
                    local_param = p
                
                # TODO: check if these need to be moved to the Dtensor if statemetn when sharding
                pseudogradient = previous_params[param_name].to(local_param.device) - local_param

                if self.skip_norm_quantization and "norm" in param_name:
                    grads[f"{param_name}_metadata"] = torch.tensor([0.0], device=local_param.device)
                    grads[f"{param_name}_values"] = pseudogradient
                    continue

                if self.skip_embedding_quantization and self._is_param_embedding(param_name):
                    grads[f"{param_name}_metadata"] = torch.tensor([0.0], device=local_param.device)
                    grads[f"{param_name}_values"] = pseudogradient
                    continue



                if self.use_ef:
                    error_buffer = self.state[p]["error_buffer"]

                    # Update the error buffer: e_t = decay * e_{t-1} + lr * g_t.
                    if self.error_decay != 1.0:
                        error_buffer.mul_(self.error_decay)
                    error_buffer.add_(pseudogradient, alpha=self.lr)

                    packed_tensor, metadata = self.compressor._quantize(error_buffer)
                    local_reconstruction = self.compressor._dequantize(packed_tensor, metadata)
                    error_buffer.sub_(local_reconstruction, alpha=self.error_feedback_alpha)

                    # Store the compressed gradient data for communication
                    grads[f"{param_name}_metadata"] = metadata
                    grads[f"{param_name}_values"] = packed_tensor
                else:
                    packed_tensor, metadata = self.compressor._quantize(pseudogradient)
                    grads[f"{param_name}_metadata"] = metadata
                    grads[f"{param_name}_values"] = packed_tensor

    def _set_grads(
            self, 
            grads: Dict[str, torch.Tensor], 
            fragment: Any,
            global_step: int = 0
        ):
        """
        Sets the gradients of the model parameters from the all-gathered compressed data.
        """
        with torch.no_grad():

            for param_name, p in fragment.named_parameters():
                # get all-gather keys
                metadata_key = f"{param_name}_metadata"
                values_key = f"{param_name}_values"

                # get locally stored information
                if metadata_key not in grads or values_key not in grads:
                    print(f"metadata_key or values_key not in grads: {metadata_key} or {values_key}")
                    continue
                
                gathered_metadata = grads[metadata_key]
                gathered_values = grads[values_key]


                if self.skip_norm_quantization and "norm" in param_name:
                    p.grad.copy_(torch.stack(gathered_values).mean(dim=0).detach())
                    del grads[metadata_key]
                    del grads[values_key]
                    continue

                if self.skip_embedding_quantization and self._is_param_embedding(param_name):
                    p.grad.copy_(torch.stack(gathered_values).mean(dim=0).detach())
                    del grads[metadata_key]
                    del grads[values_key]
                    continue

                aggregated_gradient = torch.stack([
                        self.compressor._dequantize(packed_tensor, metadata)
                        for packed_tensor, metadata in zip(gathered_values, gathered_metadata)
                    ]).mean(dim=0)


                if self.simulate_quantization_after_reduce:
                    # to simulate an efficient implementation with all-to-all reduce-scatter
                    # followed by an all-gather, we need to quantize after the reduce-scatter
                    # and dequantize after the all-gather
                    q, m = self.compressor._quantize(aggregated_gradient)
                    aggregated_gradient = self.compressor._dequantize(q, m)
                
                
                # Set the parameter's gradient to the decompressed result
                if isinstance(p, DTensor):
                    raise NotImplementedError("DTensor not supported")
                    # p.grad = DTensor.from_local(
                    #     aggregated_gradient,
                    #     p.device_mesh,
                    #     p.placements,
                    #     shape=p.shape,
                    #     stride=p.stride(),
                    # )
                else:
                    p.grad.copy_(aggregated_gradient.detach())

                del grads[metadata_key]
                del grads[values_key]



    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step."""
        if closure:
            closure()

        # 8. Perform the actual optimizer step (e.g., SGD) using the new gradient.
        super().step()


 









class WorkerGroup:
    def __init__(self, group_workers: List[int]):
        assert type(group_workers[0]) in [list, int]
        self.group_workers = group_workers

        if type(self.group_workers[0]) == list:
            self.group_workers = [sorted(x) for x in self.group_workers]
            print("sorted group workers", self.group_workers)

    def get_indices_to_avg(self):
        if type(self.group_workers[0]) == list:
            return [x[0] for x in self.group_workers] # this should already be sorted
        else:
            return self.group_workers


    def get_all_workers(self):
        if type(self.group_workers[0]) == list:
            assert type(self.group_workers[0][0]) == int
            return [x for sublist in self.group_workers for x in sublist]
        else:
            return self.group_workers







