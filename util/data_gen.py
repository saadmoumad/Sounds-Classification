from audio_preprocessing import preprocess
import pandas as pd

class Gen_Dataset(Dataset):
    def __init__(self, base_dir, meta_path, eval=False):
        self.eval = eval
        #self.base_dir = os.path.join(base_dir,'test') if self.test else os.path.join(base_dir,'train')
        #self.csv_path = os.path.join(meta_path,'test.csv') if self.test else os.path.join(meta_path,'train.csv')
        self.base_dir = base_dir
        self.csv_path = meta_path
        
        self.file_names = []
        self.labels = []
        
        self.preproces = preprocess(self.base_dir, configuration_dict.get('sample_rate'))
        self.spec_len = configuration_dict.get('spec_len')
        
        csvData = pd.read_csv(self.csv_path)
        
        self.start_indx = 114 if self.eval else 0
        self.end_indx = len(csvData) if self.eval else 114
        
    
        for i in range(self.start_indx, self.end_indx):
                self.file_names.append(csvData.iloc[i, 0])
                try:
                    self.labels.append(csvData.iloc[i, 1])  
                except AttributeError:
                    pass 
                

                
    def __getitem__(self, index):
        audio_path = os.path.join(self.base_dir,self.file_names[index]+'.wav')
        mfcc_spec = self.preproces.get_audio_MFCC(audio_path, self.spec_len, normalisation=False)
        
        #if self.test:
           # return mfcc_spec, self.file_names[index]
        
        return mfcc_spec, self.labels[index]
    
    def __len__(self):
        return len(self.file_names)