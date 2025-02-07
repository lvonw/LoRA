import argparse


def prepare_arg_parser():
    parser = argparse.ArgumentParser(prog="LoRA Test",
                                     description="Test LoRA")
    parser.add_argument("-ft",
                        "--fine-tune",
                        dest="do_fine_tune", 
                        help="",
                        action="store_true",
                        type=bool)
    
    parser.add_argument("-i",
                        "--infer",
                        dest="do_infer", 
                        help="",
                        action="store_true",
                        type=bool)
    return parser