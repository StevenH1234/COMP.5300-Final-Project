**README FILE**

Please reference Intelliscope (https://github.com/ai4colonoscopy/IntelliScope) repository to understand the model and download the material needed.
    ColonINST documentation guideline: https://github.com/ai4colonoscopy/IntelliScope/blob/main/docs/guideline-for-ColonINST.md
    ColonGPT documentation guideline: https://github.com/ai4colonoscopy/IntelliScope/blob/main/docs/guideline-for-ColonGPT.md

\
**DATASET PREPERATION**

Under the ColonINST documentation guideline all of the datasets are shown to construct the dataset. If you have permission you should download all datasets and follow the guidelines as instructed. 
Our team only downloaded the publicly available dataset which includes: BKAI-Small, CP-Child, CVC-ClinicDB, KVasic-Capsule, Kvasir-Instrument, Nerthus, and WCE-CCDD
Once you have downloaded all of the data please reference the data_reorganize.csv file which can be found here (https://drive.google.com/drive/folders/1ng2DQav-Gfts6hIr3_vCUC-a2gCWzzCO)
  - Update the file to include only the data that you have.
  - For CVC-ClinicDB data you will need to update the CSV file. Under the original column you will need to ensure that the format is labeled as a .tif file as the CSV file indicates it is a jpef file
After you have completed the above steps continue following the instructions from step 1 defined within ColonINST documentation. Please Note any step involving singularity or docker containers was replaced
with simple python virtual environment. The requirements for the virtual environment can be found in the corresponding requirements.txt file

\
**TRAINING & INFERENCE**

Once the data has been prepared according to the ColonINST documentation, proceed to the training documentation of ColonGPT (linked above). Follow the guidlines for stage 1 pre-alignment and stage 2 
fine-tuning. Instead of modifying any paths as the guidlines suggest, utilize the existing slurm files found in the scripts directory of this project. Once training is complete, 2 checkpoint directories can be found 
in the file tree. Next, follow the steps for inference. Please note that the require json files are not added directly after training is complete. Manually add the files found in the test directory at 
https://huggingface.co/datasets/ai4colonoscopy/ColonINST-v1/tree/main/test to the same directory as the second fine-tuning checkpoint. After running inference on the model, results should appear in the
same folder as text files. The results of our experiments can be found inside of the 1st sub directory of IntelliScope. 

\
**AKNOWLEDGEMENTS**

This project draws greatly upon the work done by The authors of "Frontiers in Intelligent Colonocsopy," who are also the creators of the ColonGPT model examined in this project. Their work adn findings can 
be found at https://arxiv.org/abs/2410.17241
