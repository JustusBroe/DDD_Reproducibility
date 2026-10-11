#!/usr/bin/env bash
# UCloud batch entry point for the full paper plan: ResNets 4000 epochs, CNN 500K steps,
# all 12 runs at once on one B200. Select this file under "Batch processing".
export PER_GPU=12 ADAM_EPOCHS=4000
exec bash /work/DDD_PersonB/epoch_dd/ucloud/run_epochwise.sh
