torchrun --nproc_per_node=8 lemma_exploration/parallel_finetune.py --model-name="deepseek-ai/deepseek-coder-1.3b-base" --dataset-name="yalhessi/lemexp-task1-v2" --dataset-config="template_small"  --label="ddp-8lr-v2" --training-args num_train_epochs=12 eval_steps=1/60 learning_rate=8e-4 ;
torchrun --nproc_per_node=8 lemma_exploration/parallel_finetune.py --model-name="deepseek-ai/deepseek-coder-1.3b-base" --dataset-name="yalhessi/lemexp-task1-v2" --dataset-config="template_small_notypes"  --label="ddp-8lr-v2" --training-args num_train_epochs=12 eval_steps=1/60 learning_rate=8e-4 ;
torchrun --nproc_per_node=8 lemma_exploration/parallel_finetune.py --model-name="deepseek-ai/deepseek-coder-1.3b-base" --dataset-name="yalhessi/lemexp-task1-v2" --dataset-config="template_small_nodefs"  --label="ddp-8lr-v2" --training-args num_train_epochs=12 eval_steps=1/60 learning_rate=8e-4 ;

make parallel-predictions FINETUNE_CONFIG="template_small" EVAL_CONFIG="template_small" ;
make parallel-predictions FINETUNE_CONFIG="template_small_notypes" EVAL_CONFIG="template_small_notypes" ;
make parallel-predictions FINETUNE_CONFIG="template_small_nodefs" EVAL_CONFIG="template_small_nodefs" ;


torchrun --nproc_per_node=8 lemma_exploration/parallel_finetune.py --model-name="deepseek-ai/deepseek-coder-1.3b-base" --dataset-name="yalhessi/lemexp-task1-v2" --dataset-config="lemma_object_small"  --label="ddp-8lr-v2" --training-args num_train_epochs=12 eval_steps=1/60 learning_rate=8e-4 ;  
torchrun --nproc_per_node=8 lemma_exploration/parallel_finetune.py --model-name="deepseek-ai/deepseek-coder-1.3b-base" --dataset-name="yalhessi/lemexp-task1-v2" --dataset-config="lemma_object_small_notypes"  --label="ddp-8lr-v2" --training-args num_train_epochs=12 eval_steps=1/60 learning_rate=8e-4 ;
torchrun --nproc_per_node=8 lemma_exploration/parallel_finetune.py --model-name="deepseek-ai/deepseek-coder-1.3b-base" --dataset-name="yalhessi/lemexp-task1-v2" --dataset-config="lemma_object_small_nodefs"  --label="ddp-8lr-v2" --training-args num_train_epochs=12 eval_steps=1/60 learning_rate=8e-4

make parallel-predictions FINETUNE_CONFIG="lemma_object_small" EVAL_CONFIG="lemma_object_small" ;
make parallel-predictions FINETUNE_CONFIG="lemma_object_small_notypes" EVAL_CONFIG="lemma_object_small_notypes" ;
make parallel-predictions FINETUNE_CONFIG="lemma_object_small_nodefs" EVAL_CONFIG="lemma_object_small_nodefs" 
