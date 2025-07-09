#!/bin/bash
#### ie: bash src/Bash_scripts/run_experiments.sh <data-dir> <models-dir> <output-dir> data_config/ACDC_256.yaml 1 20 train_config/train_config_200_100_00001.yaml 0
#### ie: bash src/Bash_scripts/run_experiments.sh <data-dir> <models-dir> <output-dir> data_config/ACDC_256.yaml 1 20 train_config/train_config_200_100_00001.yaml 0


# Function to handle SIGINT signal
interrupt_handler() {
    echo "Interrupted. Exiting..."
    exit 1
}

# Assign SIGINT signal to interrupt_handler function
trap interrupt_handler SIGINT


# Assign input arguments to variables
DATA_DIR=$1
MODELS_DIR=$2
OUTPUT_DIR=$3
DATA_CONFIG=$4  # ie: "data_config/ACDC_256.yaml"
CLASS_TO_SEGMENT=$5  # ie: 1 or 3
NUM_SAMPLES=$6 # ie: 0 (all samples), 10, or 20
TRAIN_CONFIG=$7 # ie: "train_config/train_config_200_100_00001.yaml" or "train_config/train_config_20_10_00001.yaml" (if $NUM_SAMPLES=0)
GPU_IDX=$8

# Logging input parameters for debugging
echo "Input Parameters: "
echo "Data Config: $DATA_CONFIG"
echo "Class to Segment: $CLASS_TO_SEGMENT"
echo "Number of Samples: $NUM_SAMPLES"
echo "Train Config: $TRAIN_CONFIG"

# Define the base command
BASE_CMD="python src/main.py"

# If using all samples
if [ $NUM_SAMPLES -eq 0 ]; then
    declare -a train_indice_list=("")
    declare -a loss_configs=("loss_config/WBCE_Dice/wbcedice_gtpromptedpred.yaml loss_config/BoxSizePrior/base_70-90_mult11_freq5_W001.yaml loss_config/BinaryCrossEntropy_OuterBoundingBox/W0001.yaml loss_config/Consistency/L2_weak_W0001.yaml")
    case $DATA_CONFIG in
        data_config/HC_640.yaml)
            # Override loss_configs for HC dataset. Using stronger and more frequent update of log barrier for size constraint
            declare -a loss_configs=("loss_config/WBCE_Dice/wbcedice_gtpromptedpred.yaml loss_config/BoxSizePrior/base_70-90_mult2_freq1_W001.yaml loss_config/BinaryCrossEntropy_OuterBoundingBox/W0001.yaml loss_config/Consistency/L2_weak_W0001.yaml")
            ;;
        data_config/MSDSpleen_512.yaml)
            # Override loss_configs for MSD Spleen. Using different augmentations for consistency loss
            declare -a loss_configs=("loss_config/WBCE_Dice/wbcedice_gtpromptedpred.yaml loss_config/BoxSizePrior/base_70-90_mult11_freq5_W001.yaml loss_config/BinaryCrossEntropy_OuterBoundingBox/W0001.yaml loss_config/Consistency/L2_stronger_W0001.yaml")
            ;;
        *)
            # If DATA_CONFIG does not match any case, use the default loss_configs (already set above)
            echo "Using default loss_configs for unknown DATA_CONFIG: $DATA_CONFIG"
            ;;
    esac

# If using 10 samples
elif [ $NUM_SAMPLES -eq 10 ]; then
    declare -a loss_configs=("loss_config/WBCE_Dice/wbcedice_gtpromptedpred.yaml loss_config/BoxSizePrior/base_70-90_mult11_freq5_W001.yaml loss_config/BinaryCrossEntropy_OuterBoundingBox/W0001.yaml loss_config/Consistency/L2_weak_W0001.yaml")
    case $DATA_CONFIG in
        data_config/ACDC_256.yaml)
            declare -a train_indice_list=("287 280 473 320 213 761 535 662 678 123" "619 596 483 557 479 722 212 125 729 243" "91 386 394 38 407 138 757 30 480 334")
            ;;
        data_config/HC_640.yaml)
            declare -a train_indice_list=("387 152 248 107 236 8 491 33 398 440" "21 506 3 418 422 143 188 9 374 68" "207 407 65 299 200 206 181 504 338 326")
            ;;
        data_config/MSDSpleen_512.yaml)
            declare -a train_indice_list=("3 38 54 59 130 214 233 258 364 450" "43 153 180 234 285 303 444 550 571 596" "187 213 259 297 310 355 366 375 395 494")
            # Override loss_configs for MSD Spleen dataset. Using different augmentations for consistency loss
            declare -a loss_configs=("loss_config/WBCE_Dice/wbcedice_gtpromptedpred.yaml loss_config/BoxSizePrior/base_70-90_mult11_freq5_W001.yaml loss_config/BinaryCrossEntropy_OuterBoundingBox/W0001.yaml loss_config/Consistency/L2_augaverage_W0001.yaml")
            ;;
        *)
            echo "Unknown DATASET configuration: $DATA_CONFIG"
            echo "Using default loss_configs for unknown DATA_CONFIG: $DATA_CONFIG"
            exit 1
            ;;
    esac

# If using 20 samples
elif [ $NUM_SAMPLES -eq 20 ]; then
    declare -a loss_configs=("loss_config/WBCE_Dice/wbcedice_gtpromptedpred.yaml loss_config/BoxSizePrior/base_70-90_mult11_freq5_W001.yaml loss_config/BinaryCrossEntropy_OuterBoundingBox/W0001.yaml loss_config/Consistency/L2_weak_W0001.yaml")
    case $DATA_CONFIG in
        data_config/ACDC_256.yaml)
            declare -a train_indice_list=("71 122 130 135 158 182 217 220 243 284 303 440 545 604 612 613 679 682 707 760" "8 157 203 216 259 330 348 366 415 449 528 543 554 575 598 608 621 680 720 750" "22 57 59 97 104 118 123 273 372 452 492 494 520 561 571 596 600 658 664 706")
            ;;
        data_config/CAMUS_512.yaml)
            declare -a train_indice_list=("11 25 51 59 60 65 95 144 151 180 199 204 208 226 231 232 233 266 276 328" "5 7 76 106 153 155 156 171 173 220 222 224 234 238 254 260 279 290 292 307" "5 25 37 55 94 103 126 129 134 136 142 157 189 193 204 213 263 274 302 335")
            ;;
        data_config/HC_640.yaml)
            declare -a train_indice_list=("11 18 20 29 65 73 98 141 193 225 291 311 314 334 335 343 364 435 439 451" "15 18 73 88 124 137 174 188 197 228 234 238 273 295 345 442 456 461 499 501" "19 42 88 133 157 185 189 264 312 332 339 361 366 373 409 416 427 476 478 485")
            ;;
        data_config/MSDSpleen_512.yaml)
            declare -a train_indice_list=("13 14 55 128 130 161 180 194 208 226 243 269 392 414 421 455 478 500 508 580" "106 140 157 164 199 214 222 287 289 296 303 317 327 388 433 440 464 477 517 590" "87 100 182 204 236 253 270 276 277 406 419 473 475 487 503 518 523 554 560 576")
            # Override loss_configs for MSD Spleen dataset. Using different augmentations for consistency loss
            declare -a loss_configs=("loss_config/WBCE_Dice/wbcedice_gtpromptedpred.yaml loss_config/BoxSizePrior/base_70-90_mult11_freq5_W001.yaml loss_config/BinaryCrossEntropy_OuterBoundingBox/W0001.yaml loss_config/Consistency/L2_stronger_W0001.yaml")
            ;;
        data_config/MSDLiver_256.yaml)
            declare -a train_indice_list=("922 1590 2447 2698 2746 2883 4863 5280 6447 6676 7099 8234 8269 9034 9511 10089 10270 10402 10811 11863" "131 538 1543 4124 6157 6454 6617 6678 6987 8026 8627 9162 9665 10780 11060 11483 12632 12710 12762 12856" "463 707 887 1081 1397 1752 2824 3080 3375 3605 4759 6649 7657 8601 8640 9517 9732 10556 11906 12792")
            # Override loss_configs for MSD Liver dataset. Using different augmentations for consistency loss
            declare -a loss_configs=("loss_config/WBCE_Dice/wbcedice_gtpromptedpred.yaml loss_config/BoxSizePrior/base_70-90_mult11_freq5_W001.yaml loss_config/BinaryCrossEntropy_OuterBoundingBox/W0001.yaml loss_config/Consistency/L2_stronger_W0001.yaml")
            ;;
        *)
            echo "Unknown DATASET configuration: $DATA_CONFIG"
            echo "Using default loss_configs for unknown DATA_CONFIG: $DATA_CONFIG"
            exit 1
            ;;
    esac

else
    echo "NUM_SAMPLES must be 0, 10, or 20."
    exit 1
fi

# Define arrays of options for each variable parameter
declare -a seeds=("0" "1" "2")

# Define fixed arguments that don't change between iterations
FIXED_ARGS="$BASE_CMD \
    --data_dir $DATA_DIR \
    --models_dir $MODELS_DIR \
    --output_dir $OUTPUT_DIR \
    --data_config $DATA_CONFIG \
    --prompt_config prompt_config/box_tight.yaml \
    --data__class_to_segment $CLASS_TO_SEGMENT \
    --model_config model_config/ours_samh_config.yaml \
    --module_config model_config/module_hardnet_config.yaml \
    --train_config $TRAIN_CONFIG \
    --gpu_idx $GPU_IDX"

# Loop through each combination of variable parameters
for seed in "${seeds[@]}"; do
    for train_indices in "${train_indice_list[@]}"; do
        for loss in "${loss_configs[@]}"; do
        # Construct variable arguments
        VAR_ARGS="--loss_config $loss --seed $seed"

        # Conditionally add --train__train_indices if it's not empty
        if [ ! -z "$train_indices" ]; then
            VAR_ARGS="$VAR_ARGS --train__train_indices $train_indices"
        fi

        # Combine fixed and variable arguments
        CMD="$FIXED_ARGS $VAR_ARGS"

        echo "Executing: $CMD"
        eval $CMD || {
            echo "Error executing command: $CMD"
            continue
        }
        done
    done
done
