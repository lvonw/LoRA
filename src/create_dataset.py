from pypdf import PdfReader
import os
import constants

source_path = os.path.join(constants.DATASETS_PATH_MASTER,
                           "Custom",
                           "SAP",
                           "source.pdf")

def main():
    reader = PdfReader(source_path)
    number_of_pages = len(reader.pages)
    page = reader.pages[19]
    text = page.extract_text()

    
    
    print (number_of_pages)
    input()
    print (page)
    input()
    print (text)




if __name__ == "__main__":
    main()
