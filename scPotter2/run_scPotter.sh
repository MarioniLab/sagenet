for seed in 1996; do
    # for tag_ref in embryo1_2; do
    #     for tag_query in atlas_8.5; do
    #         bsub -o ../../logs/scPotter/seqfish_mouse_embryo2 -q production-rh74 -M 8000 -R rusage[mem=8000]\
    #         -P gpu -gpu - "python3 scPotter/run_scPotter.py \
    #         -i ../../data_tidy --tag seqfish_mouse_embryo \
    #         --tag_ref $tag_ref \
    #         --tag_query  $tag_query \
    #         --oo ../../output \
    #         --epochs 20 \
    #         --seed $seed" 
    #     done
    # done

    for tag_ref in embryo1_2; do
       for tag_query in embryo1_2 atlas_8.5; do
           bsub -o ../../logs/scPotter/seqfish_mouse_embryo  -q production -M 8000 -R rusage[mem=8000] \
           "python3 scPotter/run_scPotter.py \
           -i ../../data_tidy --tag seqfish_mouse_embryo \
           --tag_ref $tag_ref \
           --tag_query  $tag_query \
           --oo ../../output \
          --epochs 15 \
          --train_class_col res_0.05\
          --seed $seed" 
       done
    done

    for tag_ref in embryo1_2; do
       for tag_query in embryo1_2; do
           bsub -o ../../logs/scPotter/seqfish_mouse_embryo  -q production -M 8000 -R rusage[mem=8000] \
           "python3 scPotter/run_scPotter.py \
           -i ../../data_tidy --tag seqfish_mouse_embryo \
           --tag_ref $tag_ref \
           --tag_query  $tag_query \
           --oo ../../output \
          --epochs 15 \
          --train_class_col res_0.05\
          --seed $seed\
          --imp True" 
       done
    done

    # for tag_query in CN73_C2  CN73_D2 CN73_E1 CN73_E2 CN74_C1 CN74_D1 CN74_D2 CN74_E1 CN74_E2; do
    #     bsub -o ../../logs/scPotter/visium_human_heart_loo -q production -M 2500 -R rusage[mem=2500] \
    #     "python3 scPotter/run_scPotter.py \
    #     -i ../../data_tidy --tag visium_human_heart\
    #     --tag_ref $tag_query'_lo'\
    #     --tag_query  $tag_query \
    #     --oo ../../output \
    #     --epochs 20 \
    #     --seed $seed" 
    # done

    # for tag_query in CN73_C2; do
    #     bsub -o ../../logs/scPotter/visium_human_heart_loo -q gpu -M 4000 -R rusage[mem=4000] \
    #     "python3 scPotter/run_scPotter.py \
    #     -i ../../data_tidy --tag visium_human_heart\
    #     --tag_ref $tag_query'_lo'\
    #     --tag_query  $tag_query \
    #     --oo ../../output \
    #     --epochs 20 \
    #     --seed $seed" 
    # done

   # for tag_ref in ST; do
   #     for tag_query in scRNAseq; do
   #         bsub -o ../../logs/scPotter/visium_human_heart -M 8000 -R rusage[mem=8000] \
   #         "python3 scPotter/run_scPotter.py \
   #         -i ../../data_tidy --tag visium_human_heart\
   #         --tag_ref $tag_ref \
   #         --tag_query  $tag_query \
   #         --oo ../../output \
   #         --epochs 20 \
   #         --imp True \
   #         --seed $seed" 
   #     done
   # done


   # for tag_ref in ST; do
   #     for tag_query in ST; do
   #         bsub -o ../../logs/scPotter/visium_human_heart -M 8000 -R rusage[mem=8000] \
   #         "python3 scPotter/run_scPotter.py \
   #         -i ../../data_tidy --tag visium_human_heart\
   #         --tag_ref $tag_ref \
   #         --tag_query  $tag_query \
   #         --oo ../../output \
   #         --epochs 20 \
   #         --imp True \
   #         --seed $seed" 
   #     done
   # done
done


