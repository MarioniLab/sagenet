lo = "_lo"
for seed in 1996; do
        for tag_test in CN73_C2  CN73_D2 CN73_E1 CN73_E2 CN74_C1 CN74_D1 CN74_D2 CN74_E1 CN74_E2; do
            bsub -o logs/visium_human_heart_loo -q production-rh74 -M 8000 -R rusage[mem=8000] \
            -P gpu -gpu - "python3 scPotter/run_scPotter.py \
            -i ../../data_tidy --tag visium_human_heart\
            --tag_train $tag_test'_lo'\
            --tag_test  $tag_test \
            --oo ../../output \
            --epochs 30 \
            --seed $seed" 
        done
done