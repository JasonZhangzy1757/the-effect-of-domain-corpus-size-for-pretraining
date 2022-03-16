import fitz
import os
import magic
import time
import shutil
from typing import List
from collections import defaultdict


def process_all_folders(data_dir: str='/Users/americanthinker1/NationalSecurityBERT/Data/pretraining/') -> List[dict]:
    '''
    Given a directory with multiple subdirectories filled with PDF files, 
    sequentially process each file and create a list of dicts.
    '''

    folders = os.listdir(data_dir)
    folders = [folder for folder in folders if not folder.startswith('.') and os.path.isdir(os.path.join(data_dir, folder))]
    print(f'Processing data from the following folders: {folders}')
    time.sleep(2)
    pdf_dicts = []

    start = time.perf_counter()

    for folder in folders:
        
        dir_path = os.path.join(data_dir, folder)
        files = os.listdir(dir_path)
        print(f'Processing {folder} folder of {len(files)} files')
        
        count = 0
        for pdf in files:
            count += 1
            
            try:
                #creates Document object from PDF
                path = os.path.join(dir_path, pdf)
                doc = fitz.open(path)

                #process pdf page by page and join as one string
                pages = [page.get_text() for page in doc]
                text = ' '.join(pages)

                pdf_dict = {'content':text, 'file_path': path, 'page_count':doc.page_count}
                pdf_dicts.append(pdf_dict)
                
            except Exception:
                print(Exception)
                continue
                
            if count % 100 == 0:
                print(f'Processed {count} files of {len(files)}')
                
    end = time.perf_counter() - start

    print(f'Total processing time for {len(pdf_dicts)} files: {round(end/60, 1)} minutes')

    return pdf_dicts


def process_single_folder(folder: str, data_dir: str='/Users/americanthinker1/NationalSecurityBERT/Data/pretraining/') -> List[dict]:
    '''
    Given a directory with multiple subdirectories filled with PDF files, 
    sequentially process each file and create a list of dicts.
    '''
    folder_path = os.path.join(data_dir, folder)
    files = os.listdir(folder_path)
    files = [file for file in files if file.endswith('pdf') and os.path.isfile(os.path.join(folder_path, file))]
    print(f'Processing {len(files)} files from the following folder: {folder}')
    time.sleep(2)

    pdf_dicts = []
    start = time.perf_counter()
    count = 0

    for pdf in files:
        count += 1
        
        try:
            #creates Document object from PDF
            path = os.path.join(folder_path, pdf)
            doc = fitz.open(path)

            #process pdf page by page and join as one string
            pages = [page.get_text() for page in doc]
            text = ' '.join(pages)

            pdf_dict = {'content':text, 'file_path': path, 'page_count':doc.page_count}
            pdf_dicts.append(pdf_dict)
            
        except Exception:
            print(Exception)
            continue
            
        if count % 50 == 0:
            print(f'Processed {count} files of {len(files)}')
                
    end = time.perf_counter() - start

    print(f'Total processing time for {len(pdf_dicts)} files: {round(end/60, 1)} minutes')

    return pdf_dicts