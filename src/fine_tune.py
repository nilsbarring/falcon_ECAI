import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import transformers
import loralib as lora
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from datasets import concatenate_datasets
import numpy as np
from peft import prepare_model_for_kbit_training
import pandas as pd
import argparse

# Specify the model name


model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-7b",
    load_in_8bit=True,
    device_map={"":0},
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    "tiiuae/falcon-7b",
)


model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def fine_tune():
    config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["query_key_value"],
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)


    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)


    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)


    ds = load_dataset("OpenAssistant/oasst1")
    # lets convert the train dataset to a pandas df
    df = ds["train"].to_pandas()
    df = df.sample(frac=0.25, random_state=42)
    message_tree_ids = np.unique(np.array(df["message_tree_id"]))
    messages = {}
    messages['message_tree_id'] = []
    messages['message_tree_text'] = []




    for message_tree_id in message_tree_ids:
        try:
            one_message_tree = df.query(f"message_tree_id == '{message_tree_id}'").sort_values("created_date")
            text = ""
        
            text += "<human>: " + one_message_tree.iloc[0].text

            children = one_message_tree[one_message_tree.parent_id == one_message_tree.iloc[0].message_id]

            child = children[children['rank'] == 0.0]
            text += '\n' + "<bot>: " + child.iloc[0].text
        
            flag=True
            while flag:
                try:
                    children = one_message_tree[one_message_tree.parent_id == child.message_id.iloc[0]]
                    children.index
                    one_message_tree.loc[children.index].iloc[0].role
                    text += '\n' + "<human>: " + one_message_tree.loc[children.index].iloc[0].text
        
                    children = one_message_tree[one_message_tree.parent_id == one_message_tree.loc[children.index].iloc[0].message_id]
                    #children
                    child = children[children['rank'] == 0.0]
                    text += '\n' + "<bot>: " + child.iloc[0].text
                except:
                    flag=False
        
            messages['message_tree_id'].append(message_tree_id)
            messages['message_tree_text'].append(text)

        except IndexError:
            pass

    message_df = pd.DataFrame.from_dict(messages)



    data = Dataset.from_pandas(message_df)
    data




    tokenizer.pad_token = tokenizer.eos_token

    data = data.map(lambda samples: tokenizer(samples["message_tree_text"], padding=True, truncation=True,), batched=True)
    data



    training_args = transformers.TrainingArguments(
        auto_find_batch_size=True,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        save_total_limit=4,
        logging_steps=25,
        output_dir="./outputs",
        save_strategy='epoch',
        optim="paged_adamw_8bit",
        lr_scheduler_type = 'cosine',
        warmup_ratio = 0.05,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

def run_inference():
    prompt = """<human>: What is the capital of guatemala and can you write a poeme about it? 
    <bot>:"""

    
    

    batch = tokenizer(
        prompt,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    batch = batch.to('cuda:0')
    

    with torch.cuda.amp.autocast():
        output_tokens = model.generate(
            input_ids = batch.input_ids, 
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        print(generated_text.split("<human>: ")[1].split("<bot>: ")[-1])

def main():
    parser = argparse.ArgumentParser(description='Train or run inference on the LLM Falcon 7B model.')
    parser.add_argument('--mode', type=str, help='Mode to run the script in. Options: "train", "inference"')
    args = parser.parse_args()

    if args.mode == 'train':
        print("Training the model...")
        fine_tune()
    elif args.mode == 'inference':
        print("Running inference...")
        run_inference()
    else:
        print(f'Invalid mode "{args.mode}". Options are "train" or "inference".')

if __name__ == '__main__':
    main()

