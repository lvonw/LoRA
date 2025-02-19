import constants
from inference                      import run_inference
from args                           import prepare_arg_parser

from torchtune_pipeline.fine_tuning import LoRAFinetuneRecipeSingleDevice

import torch
import os
import platform
from omegaconf import DictConfig, OmegaConf

if platform.system() == "Linux":
    from unsloth import FastLanguageModel

import logging
import util

from datasets       import load_dataset, Dataset
from config         import Configuration 
from peft           import get_peft_model, LoraConfig, PeftModel

from transformers   import (AutoTokenizer, 
                            AutoModelForCausalLM, 
                            Trainer, 
                            TrainingArguments, 
                            DataCollatorForLanguageModeling)
from trl            import SFTTrainer


from tqdm           import tqdm

from create_dataset import create_dataset, compile_dataset



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
        max_length      = 512,#1024,
        padding         = "max_length",
        truncation      = True,
        return_tensors  = "pt")
    return tokenized_text

def main():
    # Parse args
    parser  = prepare_arg_parser()
    args    = parser.parse_args()    
    
    # Need to do this because funny haha torchtune overrides the logging config
    logging.getLogger().handlers = [] 
    # set up logging
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s | %(levelname)s | %(message)s",
        datefmt = "%d-%m-%Y %H:%M:%S")
    logging.info("Setting up Configuration")

    # Set up control flow
    do_inference    = args.do_inference[0] if args.do_inference else None
    do_fine_tune    = args.do_fine_tune
    do_create_data  = args.do_generate_data
    
    use_fine_tuned  = (do_inference     == "fine-tuned"
                       or do_inference  == "ft" 
                       or do_fine_tune)
    load_existing   = use_fine_tuned and not do_fine_tune
    
    # Set up configuration
    config = Configuration(constants.CONFIG_PATH_MASTER, "default")
    if not config.load_configuration():
        return -1
    main_config = config["Main"]
    logging.info("Successfully set up Configuration")
    
    # =========================================================================
    # PREPARE MODEL
    # =========================================================================
    # Model Params ------------------------------------------------------------
    base_model_name         = main_config["base model"]
    base_model_id           = util.build_model_id(config["Models"], 
                                        base_model_name)
    if base_model_id is None:
        logging.error("Model is not supported.")
        return -1
    base_cache_dir          = util.build_base_model_path(base_model_name)

    fine_tuned_config       = main_config["Fine-Tuned Model"]
    fine_tuned_method       = fine_tuned_config["method"]
    fine_tuning_framework    = fine_tuned_config["framework"]

    # Model -------------------------------------------------------------------
    logging.info ("Preparing base model and tokenizer")
    match fine_tuning_framework:
        case "torchtune" | "PEFT":
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                cache_dir = base_cache_dir) 
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_id,
                cache_dir = base_cache_dir)
    
        case "unsloth":
            if platform.system() != "Linux":
                logging.error("Unsloth is only supported on Linux")
                return -1
            
            checkpoint_dir = None
            for directory, _, filenames in os.walk(base_cache_dir):
                for filename in filenames:
                    if filename.endswith(".safetensors"):
                        checkpoint_dir = directory

            if checkpoint_dir is None: 
                logging.error("Checkpoint directory could not be found")
                return -1

            model, tokenizer = FastLanguageModel.from_pretrained(
                #base_cache_dir,
                model_name = checkpoint_dir,
                max_seq_length = 1024,
                dtype="bf16",
                #dtype="bfloat16",
                load_in_4bit=False, 
            )

            # model = FastLanguageModel.for_inference()
        case _:
            logging.error("Framework is not supported")
    logging.info ("Successfully prepared base model")
        
    # Adapter -----------------------------------------------------------------
    # IMPORTANT: from_pretrained overrides the model as a side effect
    if use_fine_tuned:
        dataset_name            = fine_tuned_config["dataset"]
        dataset_id, dataset_path = util.build_dataset_id_and_path(
            config["Datasets"], 
            dataset_name)

        fine_tuned_dir = util.build_fine_tuned_model_path(
            base_model_name,
            dataset_name,
            fine_tuned_method,
            fine_tuning_framework)
        
        logging.info ("Preparing fine-tuned adapter")
        if load_existing:
            match fine_tuning_framework:
                case "PEFT":
                    model = PeftModel.from_pretrained(model, fine_tuned_dir)
                case "unsloth": 
                    model = PeftModel.from_pretrained(model, fine_tuned_dir)
                    model = FastLanguageModel.for_inference(model)
                case  "torchtune":
                    model = PeftModel.from_pretrained(
                        model, 
                        os.path.join(fine_tuned_dir, "epoch_0"))
                case _:
                    logging.error("Framework is not supported")
        else:
            match fine_tuning_framework:
                case "PEFT":
                    lora_config = LoraConfig(
                        r               = 8, 
                        lora_alpha      = 32, 
                        lora_dropout    = 0.1,  
                        bias            = "none",  
                        task_type       = "CAUSAL_LM",  
                    )
                    model = get_peft_model(model, lora_config)
                case "unsloth":
                    model = FastLanguageModel.get_peft_model(
                        model,
                        r=64,
                        lora_alpha=128,
                        lora_dropout=0, #0.05,
                        bias="none",
                        # use_rslora=True
                    )
                case  "torchtune":
                    pass
                case "Axolotl":
                    logging.error("Finetuning with this framework works over the command line")
                    return -1
                case _:
                    logging.error("Framework is not supported")
                    return -1
        logging.info ("Successfully prepared fine-tuned adapter")

    # =========================================================================
    # Create Dataset 
    # =========================================================================
    if do_create_data:
        create_dataset(model,
                       tokenizer,
                       config)

    # =========================================================================
    # Inference 
    # =========================================================================
    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Du bist ein KI Assistent der FH Wedel<|eot_id|><|start_header_id|>user<|end_header_id|>
    Wer ist der Dozent des Workshop Cryptography? <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    # prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    # You are a helpful assistant that is good at maths<|eot_id|><|start_header_id|>user<|end_header_id|>A bounded sequence \\( x_{0}, x_{1}, x_{2}, \\ldots \\) such that for all natural numbers \\( i \\) and \\( j \\), where \\( i \\neq j \\), the following inequality holds:\n\\[ \\left|x_{i} - x_{j}\\right| |i - j|^{a} \\geq 1 \\]<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


    if do_inference:
        response = run_inference(config["Inference"],
                                 model,
                                 tokenizer,
                                 prompt,
                                 logging)
        print (response)

    # =========================================================================
    # Fine-Tuning
    # =========================================================================
    if do_fine_tune:
        # Load and Format Dataset =============================================
        # train_data = load_dataset(
        #     dataset_id, 
        #     dataset_name, 
        #     cache_dir=dataset_path,
        #     split="train")
        
        # val_data = load_dataset(
        #     dataset_id, 
        #     dataset_name, 
        #     cache_dir=dataset_path,
        #     split="test")
        
        # print (train_data["messages"][0])
        # print (type (train_data))
        # print (train_data)

        train_data = compile_dataset()
        val_data = compile_dataset()
        
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
        

        match fine_tuning_framework:
            # Torchtune =======================================================
            case "torchtune":      
                logging.info("Reformatting Data for torchtune")
                train_data = [
                    {"tokens":  data["input_ids"], 
                    "mask":     data["attention_mask"], 
                    "labels":   data["input_ids"]}
                    for data in tqdm(train_data)
                ]
                logging.info("Successfully reformatted Data for torchtune")


                # Find chached dir, because torchtune expects a different 
                # folder natively
                checkpoint_dir = None
                for directory, _, filenames in os.walk(base_cache_dir):
                    for filename in filenames:
                        if filename.endswith(".safetensors"):
                            checkpoint_dir = directory

                if checkpoint_dir is None: 
                    logging.error("Checkpoint directory could not be found")
                    return -1

                cfg = config["Fine-Tuning"]["torchtune"]
                cfg["output_dir"]                       = fine_tuned_dir
                cfg["checkpointer"]["checkpoint_dir"]   = checkpoint_dir
                
                cfg     = OmegaConf.create(cfg.to_dict())
                recipe  = LoRAFinetuneRecipeSingleDevice(cfg=cfg)

                recipe.setup(cfg        = cfg, 
                            dataset    = train_data,
                            tokenizer  = tokenizer)
                recipe.train()
                recipe.cleanup()
                return
        
            # PEFT ============================================================
            case "PEFT" | "unsloth":
                model.print_trainable_parameters()

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
                    bf16                        = True,  
                )

                use_sft_trainer = True
                if use_sft_trainer:
                    trainer = SFTTrainer(
                        model           = model, 
                        train_dataset   = train_data,
                        eval_dataset    = val_data,
                        data_collator   = data_collator,
                        args            = training_args,
                    )
                    # from unsloth.chat_templates import train_on_responses_only
                    # trainer = train_on_responses_only(
                    # trainer,
                    # instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
                    # response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n")
                else:
                    trainer = Trainer(
                        model           = model, 
                        train_dataset   = train_data,
                        eval_dataset    = val_data,
                        data_collator   = data_collator,
                        args            = training_args,
                    )

                logging.info ("Starting fine-tuning")
                trainer.train() 
                logging.info ("Finished")
                
                logging.info ("saving weights")
                model.save_pretrained(fine_tuned_dir)
                logging.info (f"Saved adapter at {str(fine_tuned_dir)}")

            case _: 
                logging.error("Framework not supported")
                return -1 




if __name__ == "__main__":
    main()
