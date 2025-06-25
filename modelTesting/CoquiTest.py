from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# generate speech by cloning a voice using default XTTS multilingual model and settings
tts.tts_to_file(text="Technology and voice cloning. Can the system handle questions properly?" \
                " One thousand two hundred and fourty-five. We need to solve this problem" \
                " immediately! The weather outside is beautiful today.",
                file_path="outputs/output.wav", # confusing naming convention, this is actually the output file
                speaker_wav=["voices/input.wav"], # and this is the input voice file
                language="en",
                split_sentences=True
                )