#!/bin/bash
# Dataset Symlink Setup Script for MULOCO
# This script creates symbolic links to the NemotronCC dataset

set -e

# ============================================
# Check for required environment variables
# ============================================
if [ -z "$MULOCO_PATH" ]; then
    echo -e "\033[1;31mERROR: MULOCO_PATH environment variable is not set.\033[0m"
    echo "Please set it first: export MULOCO_PATH=/path/to/full_muloco_install"
    exit 1
fi

if [ -z "$DATA_PATH" ]; then
    echo -e "\033[1;31mERROR: DATA_PATH environment variable is not set.\033[0m"
    echo "Please set it to point to your NemotronCC mixed dataset directory:"
    echo "  export DATA_PATH=/path/to/nemotroncc_mixed"
    exit 1
fi

# Verify paths exist
if [ ! -d "$MULOCO_PATH" ]; then
    echo -e "\033[1;31mERROR: MULOCO_PATH directory does not exist: $MULOCO_PATH\033[0m"
    exit 1
fi

if [ ! -d "$DATA_PATH" ]; then
    echo -e "\033[1;31mERROR: DATA_PATH directory does not exist: $DATA_PATH\033[0m"
    exit 1
fi

# ============================================
# Create dataset directories
# ============================================
DATASET_DIR="$MULOCO_PATH/torchtitan/data/nemotron_cc_mixed"
echo "Creating dataset directories at: $DATASET_DIR"

mkdir -p "$DATASET_DIR/train"
mkdir -p "$DATASET_DIR/val"

# ============================================
# Create training data symlinks (95% of data)
# ============================================
echo "Creating training data symlinks..."

TRAIN_SPLITS=(
    "005.0_010.0"
    "010.0_015.0"
    "015.0_020.0"
    "020.0_025.0"
    "025.0_030.0"
    "030.0_035.0"
    "035.0_040.0"
    "040.0_045.0"
    "045.0_050.0"
    "050.0_055.0"
    "055.0_060.0"
    "060.0_065.0"
    "065.0_070.0"
    "070.0_075.0"
    "075.0_080.0"
    "080.0_085.0"
    "085.0_090.0"
    "090.0_095.0"
    "095.0_100.0"
)

for split in "${TRAIN_SPLITS[@]}"; do
    src="$DATA_PATH/split_${split}.jsonl"
    dst="$DATASET_DIR/train/split_${split}.jsonl"
    
    if [ -f "$src" ]; then
        ln -sf "$src" "$dst"
        echo "  ✓ Linked: split_${split}.jsonl"
    else
        echo -e "  \033[1;33m⚠ Source not found: $src\033[0m"
    fi
done

# ============================================
# Create validation data symlink (5% of data)
# ============================================
echo "Creating validation data symlink..."

VAL_SRC="$DATA_PATH/split_000.0_005.0.jsonl"
VAL_DST="$DATASET_DIR/val/split_000.0_005.0.jsonl"

if [ -f "$VAL_SRC" ]; then
    ln -sf "$VAL_SRC" "$VAL_DST"
    echo "  ✓ Linked: split_000.0_005.0.jsonl (validation)"
else
    echo -e "  \033[1;33m⚠ Source not found: $VAL_SRC\033[0m"
fi

# ============================================
# Summary
# ============================================
echo ""
echo -e "\033[1;32mDataset setup complete!\033[0m"
echo "  Train directory: $DATASET_DIR/train/ (19 files)"
echo "  Val directory:   $DATASET_DIR/val/ (1 file)"
echo ""
echo "Verify with: ls -la $DATASET_DIR/*/"
