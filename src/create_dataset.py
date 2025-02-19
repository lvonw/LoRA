from pypdf import PdfReader
import os
import constants
import json

from tqdm           import tqdm


from inference import run_inference

class page():
    def __init__(self):
        self.raw_text           = None

        self.class_name         = None
        self.teacher            = None
        self.appointment_count  = None
        self.frequency          = None
        self.type               = None
        self.class_type         = None
        self.hour_amount        = None
        self.ects               = None
        self.examn_type         = None
        self.language           = None
        self.media_type         = None

        self.goals              = None
        self.content            = None
        self.literature         = None

        
        return 


PROMPT = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
          "Du bist ein Generator für LLM Trainingsbeispiele im Frage Antwort Format. " 
          "Deine Antwort erfolgt ausschließlich in gültigem JSON-Format, ohne "
          "jegliche Einleitungen oder Kommentare. "
          
          "<|eot_id|><|start_header_id|>user<|end_header_id|>"
          "Generiere etwa 30 unterschiedliche Frage-Antwort-Paare zum folgenden Text. " 
          "Der Inhalt stammt aus dem Curriculum eines Masterstudiengangs in Informatik bei der FH-Wedel. " 
          
          "Deine Antwort muss exakt diesem Muster entsprechen: \n"
          " {\"Q\": \"[Fragetext1]\",  \"A\": \"[Antworttext1]\"}, \n"
          " {\"Q\": \"[Fragetext2]\",  \"A\": \"[Antworttext2]\"}, \n"

          "Ein Beispiel für eine deiner Antworten wäre: "
          "{\"Q\": \"[Wie ist die Modulnummer des Kurses Algorithmics]\", "
          "  \"A\": \"[Die Modulnummer des Kurses Algorithmics ist M003]\"},"
          
          "Der Seiteninhalt ist: "
        )



PROMPT_END =  "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"


save_folder = os.path.join(constants.DATASETS_PATH_MASTER,
                           "Custom",
                           "SAP")

source_path = os.path.join(save_folder,
                           "source.pdf")

def parse_pdf(path):
    reader = PdfReader(path)
    if not reader:
        return None
    
    parsed_pdf          = []
    finished_preamble   = False 
    for page in reader.pages:
        page_text = page.extract_text()

        if page_text.startswith("I."):
            finished_preamble = True
            parsed_pdf.append(page_text)
        
        elif finished_preamble:
            parsed_pdf[-1] = parsed_pdf[-1] + "\n" + page_text

        elif len(page_text):
            parsed_pdf.append(page_text)

    print(parsed_pdf[15])

    exit()
    return parsed_pdf

def generate_question_pairs(page_text, 
                            model,
                            tokenizer, 
                            config):
    
    prompt = PROMPT + "\n" + page_text + PROMPT_END
    # print (prompt)

    response = run_inference(config,
                             model,
                             tokenizer,
                             prompt)
    
    print (response)
    
    return response

def create_dataset(model,
                   tokenizer, 
                   config):
    parsed_pdf = parse_pdf(source_path)

    responses = []
    for page in tqdm(parsed_pdf):
        responses.append(generate_question_pairs(page,
                                model,
                                tokenizer,
                                config["Inference"]))

    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, 
                             "data.json")
    with open(save_path, "w") as f:
        json.dump(responses, f, indent=4)

    save_path = os.path.join(save_folder, 
                             "raw_data.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        for response in responses:
            f.write(response + "\n") 


system_prompt = ("Du bist ein hilfreicher KI-Assistent, der Auskunft über " 
                 "das Curriculum des Master-Studiengangs Informatik an der "
                 "FH-Wedel geben kann")


from datasets import Dataset

def helper (entry):

    result_s = {}
    result_s["role"]    = "system"
    result_s["content"] = system_prompt

    result_q = {}
    result_q["role"]    = "user"
    result_q["content"] = entry["Q"]

    result_a = {}
    result_a["role"]    = "assistant"
    result_a["content"] = entry["A"]

    return [result_s, result_q, result_a]



def compile_dataset():
    save_path = os.path.join(save_folder, 
                             "data.json")

    with open(save_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    e = {"messages": [helper(entry) for entry in data]}
    

    # print (e)
          
    compiled_dataset = Dataset.from_dict(e)

    # print (compiled_dataset)

    return compiled_dataset



def main():
    parsed_pdf = parse_pdf(source_path)

    print (parsed_pdf[0])

if __name__ == "__main__":
    main()
