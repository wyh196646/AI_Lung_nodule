from Utils import *
import pandas as pd

original_path='/data/wyh_data/影像数据EGFR1第二部分'
res=scan_base_folder(original_path)
temp=pd.DataFrame(res)
temp.to_csv('/home/wyh21/AI_Lung_node/data_processing/temp.csv')

