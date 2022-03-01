"""
@author : Tariq Rahiman
@date : 3-1-2022
Configuring Habana DL1
docker pull public.ecr.aws/habanalabs/pytorch-installer:1.10.0-ubuntu20.04-1.2.0-585
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host public.ecr.aws/habanalabs/pytorch-installer:1.10.0-ubuntu20.04-1.2.0-585
"""

# View a task & train a model
#Let’s start by printing out the first few examples of the bAbI tasks, task 1.
# display examples from bAbI 10k task 1
parlai display_data --task babi:task10k:1

#Now let’s try to train a model on it (even on your laptop, this should train fast).
# train MemNN using batch size 1 and for 5 epochs
parlai train_model --task babi:task10k:1 --model-file /tmp/babi_memnn --batchsize 1 --num-epochs 5 --model memnn --no-cuda

#Let’s print some of its predictions to make sure it’s working.
# display predictions for model save at specified file on bAbI task 1

parlai display_model --task babi:task10k:1 --model-file /tmp/babi_memnn --eval-candidates vocab

#The “eval_labels” and “MemNN” lines should (usually) match!
#Let’s try asking the model a question ourselves.
# interact with saved model
parlai interactive --model-file /tmp/babi_memnn --eval-candidates vocab