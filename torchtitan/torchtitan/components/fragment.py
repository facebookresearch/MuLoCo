# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torchtitan.tools.logging import logger

from typing import Any




def generate_llm_fqn_per_model_part(
    num_stages: int,
    num_layers: int,
    input_weight: int = 1,
    output_weight: int = 1,
) -> list[list[str]]:
    """
    Programmatically generates module names model part, focused on LLMs models.

    Args:
        num_stages: Number of pipeline stages
        num_layers: Total number of transformer layers in the model
        input_weight: Weight for input modules (tok_embeddings) in layer calculation
        output_weight: Weight for output modules (norm + output) in layer calculation

    Returns:
        List of lists containing module names for each model part

    Example:
        generate_llm_fqn_per_model_part(2, 3, input_weight=2, output_weight=2)
        treats embeddings as 2 layers and norm+output as 2 layers for distribution
    """
    if num_stages < 1:
        raise ValueError("Number of stages must be at least 1")

    if num_stages == 1:
        # Single stage gets everything
        layer_names = [f"layers.{i}" for i in range(num_layers)]
        return [["tok_embeddings"] + layer_names + ["norm", "output"]]

    # Calculate effective layers including weights
    num_effective_layers = num_layers + input_weight + output_weight

    if num_stages > num_effective_layers:
        raise ValueError(
            f"Number of stages ({num_stages}) cannot be greater than effective layers ({num_effective_layers})"
        )

    # Calculate layers per stage (distribute evenly)
    layers_per_stage = num_effective_layers // num_stages
    extra_layers = num_effective_layers % num_stages

    # Feasibility check: Ensure at least 1 layer in each PP stage
    if layers_per_stage == 0:
        raise ValueError(
            f"Configuration would result in empty stages. "
            f"With {num_stages} stages and {num_effective_layers} effective layers "
            f"(num_layers={num_layers} + input_weight={input_weight} + output_weight={output_weight}), "
            f"each stage would get {layers_per_stage} layers on average. "
            f"Reduce num_stages or increase num_layers/weights."
        )

    # Balance check: Ensure weights don't exceed minimum layers per stage
    if input_weight > layers_per_stage:
        raise ValueError(
            f"input_weight ({input_weight}) exceeds minimum layers per stage ({layers_per_stage})."
        )
    if output_weight > layers_per_stage:
        raise ValueError(
            f"output_weight ({output_weight}) exceeds minimum layers per stage ({layers_per_stage})."
        )

    module_names_per_stage = []
    current_layer = 0

    for stage_idx in range(num_stages):
        stage_modules = []

        # Calculate effective layers for this stage
        effective_layers_for_stage = layers_per_stage
        if stage_idx < extra_layers:
            effective_layers_for_stage += 1

        # First stage: handle input modules with weighting
        if stage_idx == 0:
            stage_modules.append("tok_embeddings")
            # Account for input weight in layer distribution
            remaining_layers_for_stage = effective_layers_for_stage - input_weight

            # Add transformer layers
            for _ in range(remaining_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1

        # Last stage: handle output modules with weighting
        elif stage_idx == num_stages - 1:
            # Account for output weight in layer distribution
            remaining_layers_for_stage = effective_layers_for_stage - output_weight

            # Add transformer layers
            for _ in range(remaining_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1

            # Add output modules
            stage_modules.extend(["norm", "output"])

        # Middle stages: only transformer layers
        else:
            for _ in range(effective_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1

        module_names_per_stage.append(stage_modules)

    return module_names_per_stage


def module_split(
    model: nn.Module,
    module_fqns_per_model_fragment: list[list[str]],
) -> list[nn.Module]:
    """
    This API creates fragments based on specified module names for each fragment.
    This method updates the model in place.

    Args:
        model: The complete model to be split
        module_fqns_per_model_fragment: List of lists, where each inner list contains the module names
                               that should be included in that fragment. Module names should be
                               dot-separated paths. Examples:
                               - "tok_embeddings" for token embeddings
                               - "layers.0", "layers.1" for specific transformer layers
                               - "norm" for the final normalization layer
                               - "output" for the output projection layer

    Returns:
        List of model fragments

    Example usage:
        module_fqns_per_model_fragment = [
            ["tok_embeddings", "layers.0"],     # fragment 0: embeddings + first layer
            ["layers.1", "layers.2"],           # fragment 1: middle layers
            ["norm", "output"]                  # fragment 2: final norm + output
        ]
    """

    def _build_fragment_from_modules(
        fragment_idx: int, module_names: list[str]
    ) -> nn.Module:
        fragment_model = nn.Module()
        # Create a set of modules to keep for faster lookup
        modules_to_keep = set(module_names)
        print(f"fragment {fragment_idx}: Modules to keep: {modules_to_keep}")
        for module_name, module_value in model.named_children():
            # Handle layer-like structures (e.g., "layers.0", "layers.1")
            if isinstance(module_value, (nn.ModuleDict, nn.ModuleList)):
                layers_to_keep = {
                    name.split(".", 1)[1]
                    for name in modules_to_keep
                    if name.startswith(f"{module_name}.")
                }

                if not layers_to_keep:
                    continue

                # Keep only specified layers
                if isinstance(module_value, nn.ModuleDict):
                    for layer_name in list(module_value.keys()):
                        if layer_name in layers_to_keep:
                            setattr(
                                fragment_model,
                                f"{module_name}.{layer_name}",
                                module_value[layer_name],
                            )
                else:
                    indices_to_keep = {
                        int(idx) for idx in layers_to_keep if idx.isdigit()
                    }
                    new_layers = nn.ModuleList(
                        [
                            layer
                            for i, layer in enumerate(module_value)
                            if i in indices_to_keep
                        ]
                    )
                    setattr(fragment_model, module_name, new_layers)

                continue

            # Handle simple module attributes (e.g., "linear", "norm")
            if module_name not in modules_to_keep:
                continue

            setattr(fragment_model, module_name, module_value)

        return fragment_model

    num_fragments = len(module_fqns_per_model_fragment)
    model_fragments = []

    for fragment_idx in range(num_fragments):
        module_names = module_fqns_per_model_fragment[fragment_idx]
        model_fragment = _build_fragment_from_modules(
            fragment_idx,
            module_names,
        )
        logger.info(
            f"building fragment_idx {fragment_idx} " f"with modules {module_names}"
        )
        model_fragments.append(model_fragment)

    return model_fragments


def fragment_llm(
    model: nn.Module,
    ft_config: Any,
    n_layers: int,
) -> list[nn.Module]:
    assert ft_config.num_fragments > 0

    module_fqns_per_model_fragment = ft_config.module_fqns_per_model_fragment

    input_weight = 1  # Weight for tok_embeddings
    output_weight = 1  # Weight for norm + output layers

    if module_fqns_per_model_fragment == []:
        if ft_config.num_fragments == 1:
            logger.info("Created 1 model fragments")
            return [model]

        module_fqns_per_model_fragment = generate_llm_fqn_per_model_part(
            ft_config.num_fragments, n_layers, input_weight, output_weight
        )

    model_fragments = module_split(model, module_fqns_per_model_fragment)
    logger.info(f"Created {len(model_fragments)} model fragments")

    return model_fragments