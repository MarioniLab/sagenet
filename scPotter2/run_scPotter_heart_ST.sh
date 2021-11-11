for seed in 1996; do
    for tag_tarin in ST; do
        for tag_test in scRNAseq; do
            bsub -o logs/visium_human_heart -q production-rh74 -M 8000 -R rusage[mem=8000] \
            -P gpu -gpu - "python3 scPotter/run_scPotter.py \
            -i ../../data_tidy --tag visium_human_heart\
            --tag_train $tag_tarin \
            --tag_test  $tag_test \
            --oo ../../output \
            --epochs 30 \
            --seed $seed" 
        done
    done
done