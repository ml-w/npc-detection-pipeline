import pandas as pd
from pytorch_med_imaging.utils.batchgenerator import GenerateTestBatch
from pytorch_med_imaging.pmi_data import ImageDataSet
from pathlib import Path
from rAIdiologist.config.rAIdiologistCFG import data_loader
import os


data_dir = Path('../NPC_Segmentation/60.Large-Study/v1-All-Data/Normalized_2/T2WFS_TRA/01.NyulNormalized/',)
out_file_dir = Path('../NPC_Segmentation/99.Testing/NPC_Screening/rai_v5.1/')
ground_truth = Path('../NPC_Segmentation/99.Testing/NPC_Screening/v1/Datasheet_v2.csv')

# Read existing data
regex = r"^[\d\w]+"
images = ImageDataSet(data_dir.__str__(), verbose=True, idGlobber=regex)
im_ids =images.get_unique_IDs(regex)
gt = pd.read_csv(ground_truth, index_col=0)

# Get intersection
targets = gt.index.intersection(im_ids)
dropped = gt.index.difference(im_ids)
print(f"Dropped: {dropped}")

GenerateTestBatch(targets,
                  5,
                  out_file_dir.__str__(),
                  stratification_class=gt.loc[targets]['is_malignant'],
                  validation=len(images) // 10,
                  prefix='B'
                  )
