"""

├───.env
├───data
│   ├───test_data
│   │   ├───1000_line
│   │   │   ├───combine_data_1000.py
│   │   │   ├───ground_truth_1000.txt
│   │   │   ├───results
│   │   │   │   ├───output_darija_enhanced_1000.txt
│   │   │   │   ├───output_darija_nllb_1000.txt
│   │   │   │   └───output_darija_seamless_1000.txt
│   │   │   └───test_dataset_darija_1000.txt
│   │   ├───8000_line
│   │   │   ├───combine_data_8000.py
│   │   │   ├───ground_truth_8000.txt
│   │   │   ├───results
│   │   │   │   ├───output_darija_nllb_8000.txt
│   │   │   │   └───output_darija_seamless_8000.txt
│   │   │   └───test_dataset_darija_8000.txt
│   │   ├───output_synonyms.txt
│   │   └───results
│   │       ├───combined_output_1000.csv
│   │       └───combined_output_8000.csv
│   └───training_data
│       ├───processed
│       │   ├───ground_truth.csv
│       │   └───output_sentences.txt
│       ├───process_functions
│       │   ├───darija_sentence_extraction.py
│       │   └───taoeba_process_dataset.py
│       ├───raw
│       │   ├───DODa_sentences.csv
│       │   ├───MTCD_tweets_dataset.csv
│       │   └───taoeba_eng_sentences.tsv
│       └───references.txt
├───evaluation
│   ├───consistency_scores.csv
│   ├───model_agreements.csv
│   ├───photos
│   │   ├───BoxPlot.png
│   │   ├───Consistency_Scores_Distrubution.png
│   │   └───Frequency_Model_Agreement.png
│   └───translation_comparison_cosine_results.csv
├───models
│   ├───nllb
│   │   ├───lang_list_extended.py
│   │   ├───nllb_model.py
│   │   ├───output_nllb.txt
│   │   └───test_dataset_darija_8000.txt
│   └───seamless
│       ├───output_seamless.txt
│       ├───seamless_model.py
│       ├───test.txt
│       └───test_dataset_darija_8000.txt
├───notebooks
│   ├───evaluation.ipynb
│   ├───generate_translations.ipynb
│   ├───get_cosine_similarities.ipynb
│   └───results
│       └───final_translations_1000.csv
└───tree.py


"""

import os

def show_tree(path, indent=""):
    try:
        items = os.listdir(path)
    except PermissionError:
        return

    for index, item in enumerate(items):
        item_path = os.path.join(path, item)
        if index == len(items) - 1:
            print(f"{indent}└───{item}")
            new_indent = indent + "    "
        else:
            print(f"{indent}├───{item}")
            new_indent = indent + "│   "

        if os.path.isdir(item_path):
            show_tree(item_path, new_indent)

# Replace '/path/to/Darija' with the actual path to your Darija folder
show_tree('.')