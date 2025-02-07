import constants
import logging
import os
import torch
import util

from datasets       import load_dataset, Dataset
from config         import Configuration 
from peft           import get_peft_model, LoraConfig, PeftModel
from transformers   import (AutoTokenizer, 
                            AutoModelForCausalLM, 
                            pipeline as pl,
                            Trainer, 
                            TrainingArguments, 
                            DataCollatorForLanguageModeling)
from tqdm           import tqdm

from llama_models.llama3.reference_impl.generation import Llama
from llama_models.llama3.api.datatypes import RawMessage, StopReason

# def setup_distributed():
#     os.environ["RANK"] = "0"          
#     os.environ["WORLD_SIZE"] = "1"     
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "29500"     
#     global_rank  = int(0)
#     local_rank   = int(0)    

#     distributed.scatter_object_list

#     backend = "GLOO"
#     distributed.init_process_group(backend)
#     torch.cuda.set_device(local_rank)


def build_model_id(model_config, model_name):
    for family, members in model_config.items():
        if model_name in members:
            return family + "/" + model_name
    return None

def build_base_model_path(model_name):
    return os.path.join(constants.MODEL_PATH_BASE, model_name)

def build_fine_tuned_model_path(model_name, dataset_name, method, framework):
    return os.path.join(constants.MODEL_PATH_FINE_TUNED, 
                        model_name,
                        dataset_name,
                        method,
                        framework)

def build_dataset_id_and_path(dataset_config, dataset_name):
    for family, members in dataset_config.items():
        for member, dataset in members.items():
            if dataset_name in dataset:
                data_id     = family + "/" + member
                data_path   = os.path.join(constants.DATASETS_PATH_MASTER, 
                                           member, 
                                           dataset_name)
                return data_id, data_path 
    return None

def tokenize_function(entry, tokenizer):
    formatted_text = tokenizer.apply_chat_template(
        entry["messages"], 
        tokenize=False)     
    tokenized_text = tokenizer(
        text            = formatted_text,
        max_length      = 512,
        padding         = "max_length",
        truncation      = True,
        return_tensors  = "pt")
    return tokenized_text



def main():
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s | %(levelname)s | %(message)s",
        datefmt = "%d-%m-%Y %H:%M:%S")

    logging.info("Setting up Configuration")
    config = Configuration(constants.CONFIG_PATH_MASTER, "default")
    if not config.load_configuration():
        return -1
    logging.info("Successfully set up Configuration")

    
    # =========================================================================
    # Inference 
    # =========================================================================
    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful assistant that is good at maths<|eot_id|><|start_header_id|>user<|end_header_id|>
    "In a fruit salad, there are raspberries, green grapes, and red grapes. There are three times the number of red grapes as green grapes, plus some additional red grapes. There are 5 less raspberries than green grapes. There are 102 pieces of fruit in the salad, and there are 67 red grapes in the salad. How many more red grapes are there than three times the number of green grapes?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    do_inference = True
    if do_inference:
        inference_config = config["Inference"]
        # Generate Method ===================================================== 
        do_generate = True
        if do_generate:
            
            base_model_name  = inference_config["base model"]
            base_model_id    = build_model_id(config["Models"], 
                                                base_model_name)
            if base_model_id is None:
                logging.error("Model is not supported.")
                return -1
            base_cache_dir = build_base_model_path(base_model_name)

            logging.info ("Preparing base model")
            inference_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                cache_dir = base_cache_dir)
            logging.info ("Successfully prepared base model")
                
            logging.info ("Preparing Tokenizers")
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_id,
                cache_dir = base_cache_dir)
            logging.info ("Successfully prepared Tokenizers")

            use_fine_tuned_model = True
            if use_fine_tuned_model:
                fine_tuned_config      = inference_config["Fine-Tuned Model"]
                fine_tuned_method      = fine_tuned_config["method"]
                fine_tuned_framework   = fine_tuned_config["framework"]
                
                dataset_name = fine_tuned_config["dataset"]
                dataset_id, dataset_path = build_dataset_id_and_path(
                    config["Datasets"], 
                    dataset_name)

                # Prep fine-tuned Model Vars ==================================
                fine_tuned_dir = build_fine_tuned_model_path(
                    base_model_name,
                    dataset_name,
                    fine_tuned_method,
                    fine_tuned_framework)
                
                # This can change the inference model merely as a sideeffect
                # so in order to be safe, we explicitly override it.
                inference_model = PeftModel.from_pretrained(inference_model, 
                                                             fine_tuned_dir)



            tokenized_input = tokenizer(prompt, return_tensors="pt")
            input_ids       = tokenized_input.input_ids
            attention_mask  = tokenized_input.attention_mask

            device          = util.get_device(config = config)
            inference_model = inference_model.to(device) 
            input_ids       = input_ids.to(device)
            attention_mask  = attention_mask.to(device)
            logging.info (f"Moved model to {device}")


            output_sequences = inference_model.generate(
                input_ids, 
                max_new_tokens          = 500, 
                attention_mask          = attention_mask,
                do_sample               = False, 
                temperature             = None if True else 0.6,
                top_k                   = None if True else 10,
                top_p                   = None if True else 0.9,
                num_return_sequences    = 1,
                pad_token_id            = tokenizer.eos_token_id,
                eos_token_id            = tokenizer.eos_token_id)

            output_sequences = output_sequences.to(
                util.get_device(idle = True, config = config["Devices"]))

            output_sequences = output_sequences[:, input_ids.shape[1]:]

            response = tokenizer.decode(
                output_sequences[0], 
                skip_special_tokens=inference_config["skip special tokens"])

            print (response)


        # Pipeline Method =====================================================
        do_pipeline = False
        if do_pipeline:
            pipeline = pl(
                task            = "text-generation",
                model           = base_model,
                torch_dtype     = torch.bfloat16,
                device_map      = "auto",
                tokenizer       = tokenizer,
                model_kwargs    = {"cache_dir": base_cache_dir}
            )

            sequences = pl(
                prompt, #"def print_hello_world():",        
                do_sample               = True,
                top_k                   = 10,
                num_return_sequences    = 1,
                pad_token_id            = tokenizer.eos_token_id,
                bos_token_id            = tokenizer.bos_token_id,
                eos_token_id            = tokenizer.eos_token_id,
                truncation              = True,
                max_new_tokens          = 100,
            )

            for seq in sequences:
                print(f"Result: {seq['generated_text']}")

    # =========================================================================
    # Fine-Tuning
    # =========================================================================
    do_fine_tune = False 
    if do_fine_tune:
        fine_tuning_config      = config["Fine-Tuning"]
        fine_tuning_method      = fine_tuning_config["method"]
        fine_tuning_framework   = fine_tuning_config["framework"]

        #  Prep Base Model Vars ===============================================  
        base_model_name  = fine_tuning_config["base model"]
        base_model_id    = build_model_id(config["Models"], base_model_name)
        if base_model_id is None:
            logging.error("Base Model is not supported.")
            return -1
        base_cache_dir = build_base_model_path(base_model_name)
   
        # Prep Dataset Vars ===================================================       
        dataset_name = fine_tuning_config["dataset"]
        dataset_id, dataset_path = build_dataset_id_and_path(
            config["Datasets"], 
            dataset_name)

        # Prep fine-tuned Model Vars ==========================================
        fine_tuned_dir = build_fine_tuned_model_path(base_model_name,
                                                     dataset_name,
                                                     fine_tuning_method,
                                                     fine_tuning_framework)

        # Create Tokenizer ====================================================
        logging.info ("Preparing Tokenizers")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            cache_dir = base_cache_dir)
        logging.info ("Successfully prepared Tokenizers")

        print (dataset_name)
        print (dataset_id)
        print (dataset_path)
        print (fine_tuned_dir)

        # Load and Format Dataset =============================================
        train_data = load_dataset(
            dataset_id, 
            dataset_name, 
            cache_dir=dataset_path,
            split="train")
        
        val_data = load_dataset(
            dataset_id, 
            dataset_name, 
            cache_dir=dataset_path,
            split="test")
        
        tokenizer.pad_token = tokenizer.eos_token
        
        train_data = train_data.map(
            tokenize_function, 
            batched     = True,
            desc        = "Formatting Training Data",
            fn_kwargs   = {
                "tokenizer" : tokenizer})
        
        val_data = val_data.map(
            tokenize_function, 
            batched     = True,
            desc        = "Formatting Validation Data",
            fn_kwargs   = {
                "tokenizer" : tokenizer})
    

        # Create Base Model ===================================================
        logging.info ("Preparing models")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            cache_dir = base_cache_dir)
        logging.info ("Successfully prepared models")

        # PEFT ================================================================
        lora_config = LoraConfig(
            r               = 8, 
            lora_alpha      = 32, 
            lora_dropout    = 0.1,  
            bias            = "none",  
            task_type       = "CAUSAL_LM",  
        )

        peft_model = get_peft_model(base_model, lora_config)
        peft_model.print_trainable_parameters()

        data_collator = DataCollatorForLanguageModeling(
            tokenizer, 
            mlm=False)

        training_args = TrainingArguments(
            output_dir                  = fine_tuned_dir,
            num_train_epochs            = 1,
            per_device_train_batch_size = 1,
            per_device_eval_batch_size  = 1,
            eval_strategy               = "epoch",
            logging_dir                 = "./logs",
            logging_steps               = 500,
            save_steps                  = 500,
            save_total_limit            = 2,
        )

        trainer = Trainer(
            model           = peft_model, 
            train_dataset   = train_data,
            eval_dataset    = val_data,
            data_collator   = data_collator,
            args            = training_args,
        )

        logging.info ("Starting fine-tuning")
        trainer.train() 
        logging.info ("Finished")
        
        logging.info ("saving weights")
        peft_model.save_pretrained(fine_tuned_dir)
        logging.info (f"Saved weights adapter at {str(fine_tuned_dir)}")





if __name__ == "__main__":
    main()
