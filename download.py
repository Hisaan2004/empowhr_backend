from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

model_name = "facebook/wav2vec2-large-robust-ft-swbd-300h"

processor = Wav2Vec2Processor.from_pretrained(model_name)
processor.save_pretrained("models/wav2vec2-large-robust")

model = Wav2Vec2ForCTC.from_pretrained(model_name)
model.save_pretrained("models/wav2vec2-large-robust")