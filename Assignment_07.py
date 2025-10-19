# Step 1: Install required packages (run in terminal or notebook)
# pip install transformers torch IPython
# Step 2: Import required libraries
from transformers import VitsModel, AutoTokenizer
import torch
from IPython.display import Audio
import soundfile as sf

# Step 3: Clone and load the pre-trained TTS model from Hugging Face
model = VitsModel.from_pretrained("facebook/mms-tts-vie") # You may replace

tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")

# Step 4: Prepare input text
text = "Tôi là Thuấn , Xin chào anh em đến với bài tập của khoá AI Application Engineer"

# Step 5: Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")
# Step 6: Perform inference to generate the waveform
with torch.no_grad():
 output = model(**inputs).waveform
# Step 7: Play the generated audio in Jupyter Notebook
Audio(output.numpy(), rate=model.config.sampling_rate)
# Optional: Save audio to file (requires soundfile)
# Make saving robust: print shapes/dtype and force float32, mono/stereo shape
wav = output.detach().cpu().numpy()
print("DEBUG: waveform shape:", wav.shape, "dtype:", wav.dtype)
sr = getattr(model.config, 'sampling_rate', 22050)

# If waveform is (batch, time, channels) or (time,) or (channels, time), try to normalize
if wav.ndim == 3:
	# common shape from some TTS: (batch, channels, time) or (batch, time, channels)
	# try to reduce batch dim
	if wav.shape[0] == 1:
		wav = wav[0]
	else:
		# mix down multiple batch entries by taking first
		wav = wav[0]

if wav.ndim == 2:
	# If shape is (channels, time) transpose to (time, channels)
	if wav.shape[0] <= 2 and wav.shape[0] < wav.shape[1]:
		wav = wav.T

# Ensure float32
wav = wav.astype('float32')
print("DEBUG: post-processed waveform shape:", wav.shape, "dtype:", wav.dtype, "sr:", sr)

try:
	# explicit format and subtype to avoid libsndfile guessing issues
	sf.write('output.wav', wav, sr, format='WAV', subtype='PCM_16')
	print("Saved output.wav successfully")
except Exception as e:
	print("Failed to write output.wav:", type(e).__name__, e)
	# As a fallback, try writing raw numpy using numpy.save for debugging
	import numpy as _np
	_np.save('output_debug.npy', wav)
	print('Wrote output_debug.npy for inspection')