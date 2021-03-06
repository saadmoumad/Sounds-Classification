import torchaudio
import torch

class preprocess():
    
    def __init__(self, base_dir, sr):
        self.base_dir = base_dir
        self.sr = sr
        self.resampler = torchaudio.transforms.Resample(new_freq=48_000)
        self.MFCC = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=40)
        
    def _read_audio(self, audio_path, mono_channel=False):
        #os.path.join(self.base_dir, audio_name+".wav")
        soundData, sample_rate = torchaudio.load(audio_path, out = None, normalization = True)
        if mono_channel == True:
            soundData = torch.mean(soundData, dim=0, keepdim=True)
        if sample_rate != self.sr:
            self.resampler.orig_freq = sample_rate
            soundData = self.resampler(soundData)
        return soundData
    
    def _spec_normalisation(self, spec, eps=1e-6):
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
        spec_scaled = spec_scaled.type(torch.uint8)
        return spec_scaled
    
    def get_audio_MFCC(self, audio_path, fixed_length, normalisation=False):
        soundData = self._read_audio(audio_path=audio_path, mono_channel=True)
        mfcc_spec = self.MFCC(soundData)
        if mfcc_spec.shape[2]< fixed_length: #Maybre extreme approche
            mfcc_spec = torch.nn.functional.pad(mfcc_spec, (0, fixed_length - mfcc_spec.shape[2]))
        if normalisation==True:
            mfcc_spec = self._spec_normalisation(mfcc_spec)
        return mfcc_spec