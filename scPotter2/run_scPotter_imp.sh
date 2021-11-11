for seed in 1996; do
    for tag_ref in embryo1_2; do
        for tag_query in embryo1_5; do
            bsub -o ../../logs/scPotter/seqfish_mouse_embryo2 -q production-rh74 -M 8000 -R rusage[mem=8000]\
            -P gpu -gpu - "python3 scPotter/run_scPotter.py \
            -i ../../data_tidy --tag seqfish_mouse_embryo \
            --tag_ref $tag_ref \
            --tag_query  $tag_query \
            --oo ../../output \
            --epochs 30 \
            --seed $seed \
            --imp True" 
        done
    done


   # for tag_ref in ST; do
   #     for tag_query in CN73_C2; do
   #         bsub -o ../../logs/scPotter/visium_human_heart -q production-rh74 -M 8000 -R rusage[mem=8000] \
   #         -P gpu -gpu - "python3 scPotter/run_scPotter.py \
   #         -i ../../data_tidy --tag visium_human_heart\
   #         --tag_ref $tag_ref \
   #         --tag_query  $tag_query \
   #         --oo ../../output \
   #         --epochs 30 \
   #         --seed $seed \
   #         --imp True" 
   #     done
   # done
    
done
