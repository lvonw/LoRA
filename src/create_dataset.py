from pypdf import PdfReader
import os
import constants
import json

from tqdm           import tqdm


from inference import run_inference

table_section_keywords = ["Studiengang", 
                          "Kürzel", 
                          "Bezeichnung", 
                          "Lehrveranstaltung(en)",
                          "Verantwortliche(r)",
                          "Zuordnung zum Curriculum",
                          "Verwendbarkeit",
                          "Semesterwochenstunden",
                          "ECTS",
                          "Voraussetzungen",
                          "Dauer",
                          # Submodule
                          "Lehrveranstaltung",
                          "Dozent(en)",
                          "Hörtermin",
                          "Häufigkeit",
                          "Art",
                          "Lehrform",
                          "Semesterwochenstunden",
                          "ECTS",
                          "Prüfungsform",
                          "Sprache",
                          "Lehr- und Medienform(en)",
                          ]

content_keywords = ["Lernziele",
                    "Inhalt",
                    "Literatur",
                    ]


def starts_with_any(string, keywords):
    for k in keywords:
        if string.startswith(k):
            return k

    return ""

class Page():
    def __init__(self, raw_text):
        self.raw_text           = raw_text

        self.name = ""
        self.is_submodule = False

        self.table_paragraphs = []
        self.content_paragraphs = []

        latest_access = None

        for line in raw_text.splitlines():
            # print (line)
            parsed_line = ""

            if line.startswith ("I."):

                self.name = " ".join(line.split(" ")[1:])
                self.is_submodule = line.split()[0].count(".") >= 3
                continue
            
            keyword = starts_with_any(line, table_section_keywords)
            if keyword:
                parsed_line = keyword + ":" + line[len(keyword):]
                self.table_paragraphs.append(parsed_line)
                latest_access = self.table_paragraphs
                continue

            keyword = starts_with_any(line, content_keywords)
            if keyword:
                parsed_line = keyword + ":" + line[len(keyword):]
                self.content_paragraphs.append(parsed_line)
                latest_access = self.content_paragraphs
                continue

            if latest_access:
                latest_access[-1] += "\n" + line


    def __str__(self):
        return self.name + "\n" + str(self.table_paragraphs) + "\n" + str(self.content_paragraphs)


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

    prepared_pages = []
    for page in parsed_pdf:
        prepared_pages.append(Page(page))

    return prepared_pages







SYS_PROMPT = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
              "Du bist ein Generator für LLM Trainingsbeispiele im Frage Antwort "
              "Format anhand von Textausschnitten. Du fokussierst dich dabei "
              "stets nur auf Informationen, die Du dem gegebenen Text entnehmen kannst." 
              "Deine Antwort erfolgt ausschließlich in gültigem JSON-Objekt-Format, ohne "
              "jegliche Einleitungen oder Kommentare."
              "<|eot_id|>")


PROMPT_START ="<|eot_id|><|start_header_id|>user<|end_header_id|>"

EXAMPLE_PROMPT = ("Deine Antwort muss exakt diesem Muster entsprechen: \n"
                  " {\"Q\": \"[Fragetext1]\", \"A\": \"[Antworttext1]\"}, \n"
                  " {\"Q\": \"[Fragetext2]\", \"A\": \"[Antworttext2]\"}, \n"

                  "Beispielsweise sei dies ein gegebener Ausschnitt aus dem Kapitel Algorithmics: \n "
                  "Kürzel: M003 \n"
                  "Ein Beispiel für ein gültiges Frage-Antwort-Paar ist: \n"
                  "{\"Q\": \"[Wie ist die Modulnummer des Kurses Algorithmics?]\", "
                  " \"A\": \"[Die Modulnummer des Kurses Algorithmics ist M003.]\"},\n"
        )
PROMPT_END =  "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

def build_user_prompt(page, paragraph, amount):
    user_prompt = PROMPT_START

    user_prompt += f"Der folgende Inhalt ist ein Ausschnitt aus der Beschreibung des Kurses {page.name}"    
    user_prompt += f" aus dem Modulhandbuch der FH-Wedel. Generiere zu ihm {amount} "
    user_prompt += "Frage-Antwort Paare, welche ausschließlich aus dem Inhalt des "
    user_prompt += "gegebenen Ausschnitts entnommen werden. \n"
    user_prompt += f"Erwähne in allen deinen Fragen immer auch den Namen des aktuellen Kurses: {page.name}\n"

    user_prompt += EXAMPLE_PROMPT

    user_prompt += "Der Ausschnitt ist: \n"
    user_prompt += paragraph
    user_prompt += PROMPT_END

    return user_prompt


def generate_question_pairs(page, 
                            model,
                            tokenizer, 
                            config):
    
    question_answer_pairs = []
    for table_content in page.table_paragraphs:
        prompt = build_user_prompt(page, table_content, 4)
        # print (prompt)

        response = run_inference(config,
                                model,
                                tokenizer,
                                prompt)
        
        print (response)
        print ("------")
        
        question_answer_pairs.append(response)

    for table_content in page.content_paragraphs:
        prompt = build_user_prompt(page, table_content, 10)
        # print (prompt)

        response = run_inference(config,
                                model,
                                tokenizer,
                                prompt)
        
        question_answer_pairs.append(response)


    
    # print (response)
    
    return question_answer_pairs

def create_dataset(model,
                   tokenizer, 
                   config):
    parsed_pdf = parse_pdf(source_path)

    responses = []
    for page in tqdm(parsed_pdf[7:]):
        responses.append(generate_question_pairs(page,
                         model,
                         tokenizer,
                         config["Inference"]))
        
    
    # exit()
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
