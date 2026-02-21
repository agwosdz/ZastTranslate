import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from modules.utils import cleanup_model
from config import DEVICE


class Transcriber:
    def __init__(self, model_size="large-v3", compute_type="float16"):
        self.model_size = model_size
        self.device = DEVICE
        self.torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        self.pipe = None
        self.model = None
        self.processor = None

    def _load_model(self, model_size=None):
        """Load or reload model if size changed."""
        if model_size and model_size != self.model_size:
            self.cleanup()
            self.model_size = model_size

        if self.model is not None:
            return

        model_id = f"openai/whisper-{self.model_size}"
        print(f"Loading Whisper {self.model_size} on {self.device}...")

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def _detect_language(self, audio_path):
        """Detect language from first 30 seconds of audio."""
        try:
            audio, _ = librosa.load(audio_path, sr=16000, duration=30)
            input_features = self.processor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(self.device, dtype=self.torch_dtype)

            with torch.no_grad():
                predicted_ids = self.model.generate(input_features, max_new_tokens=1)

            # Decode the language token: "<|en|>" -> "en"
            lang_token = self.processor.batch_decode(predicted_ids[:, 1:2])[0]
            detected = lang_token.strip("<|>").strip()
            if detected:
                print(f"Detected language: {detected}")
                return detected
        except Exception as e:
            print(f"Language detection failed: {e}")
        return "en"

    @staticmethod
    def _merge_words_to_segments(chunks, max_gap=0.5, max_segment_duration=15.0):
        """
        Merge word-level timestamps into sentence-like segments.

        Groups consecutive words into segments, splitting when:
        - There's a gap > max_gap seconds between words
        - A sentence-ending punctuation is found (. ! ?)
        - Segment duration exceeds max_segment_duration

        This preserves the precise word-level boundaries from Whisper
        while producing natural sentence-level segments for dubbing.
        """
        if not chunks:
            return []

        segments = []
        current_words = []
        current_start = None

        for chunk in chunks:
            ts = chunk.get("timestamp", (None, None))
            word_start = ts[0] if ts[0] is not None else None
            word_end = ts[1] if ts[1] is not None else word_start
            text = chunk.get("text", "").strip()

            if not text or word_start is None:
                continue

            # Decide whether to start a new segment
            if current_words:
                last_end = current_words[-1]["end"]
                gap = word_start - last_end if last_end is not None else 0
                duration = word_start - current_start if current_start is not None else 0
                last_text = current_words[-1]["text"]

                should_split = (
                    gap > max_gap
                    or duration > max_segment_duration
                    or last_text.rstrip().endswith((".", "!", "?", "。", "！", "？"))
                )

                if should_split:
                    # Flush current segment
                    seg_text = " ".join(w["text"] for w in current_words).strip()
                    if seg_text:
                        segments.append({
                            "start": round(current_start, 3),
                            "end": round(current_words[-1]["end"], 3),
                            "text": seg_text,
                        })
                    current_words = []
                    current_start = None

            # Add word to current segment
            if current_start is None:
                current_start = word_start
            current_words.append({"text": text, "start": word_start, "end": word_end})

        # Flush remaining words
        if current_words:
            seg_text = " ".join(w["text"] for w in current_words).strip()
            if seg_text and current_start is not None:
                segments.append({
                    "start": round(current_start, 3),
                    "end": round(current_words[-1]["end"], 3),
                    "text": seg_text,
                })

        return segments

    def transcribe(self, audio_path, language=None, model_size=None, enable_diarization=True):
        """
        Transcribe audio with HuggingFace Whisper pipeline.
        Uses word-level timestamps for precise segment boundaries.
        Returns {"language": str, "segments": list}
        """
        self._load_model(model_size)

        # Detect language if not specified
        detected_lang = language if language else self._detect_language(audio_path)

        print("Transcription in progress...")
        generate_kwargs = {"task": "transcribe", "language": detected_lang}

        result = self.pipe(
            audio_path,
            return_timestamps="word",
            generate_kwargs=generate_kwargs,
            chunk_length_s=30,
            batch_size=16,
        )

        # Merge word-level timestamps into sentence-like segments
        segments = []
        if "chunks" in result:
            segments = self._merge_words_to_segments(result["chunks"])

        print(f"Transcription complete: {len(segments)} segments")
        return {"language": detected_lang, "segments": segments}

    def cleanup(self):
        cleanup_model(self.model)
        self.model = None
        self.processor = None
        self.pipe = None


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        t = Transcriber(model_size="base")
        res = t.transcribe(sys.argv[1], enable_diarization=False)
        print(f"Language: {res['language']}")
        for s in res['segments']:
            print(f"{s['start']}-{s['end']}: {s['text']}")
