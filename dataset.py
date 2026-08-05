import os
import json
import csv
import re
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from config import (
    PATH_TO_RAW_VIDEO,
    PATH_TO_LABEL,
    PATH_TO_TRANSCRIPTIONS,
    VIDEO_EXTENSIONS,
    TESTSET_JSON
)


class VideoDataset:
    def __init__(self, dataset_names: List[str] = None):
        if dataset_names is None:
            self.dataset_names = list(PATH_TO_RAW_VIDEO.keys())
            for dataset_name in TESTSET_JSON:
                if dataset_name not in self.dataset_names:
                    self.dataset_names.append(dataset_name)
        else:
            self.dataset_names = dataset_names
        self._video_cache = {}

    def _parse_json_prompt_as_transcription(self, dataset_name: str, prompt: str) -> str:
        if dataset_name == 'Openr1psy':
            return prompt

        if dataset_name == 'AvaMERG':
            try:
                from prompts.erg import get_erg_prompt
                base_prompt = get_erg_prompt()
                if prompt.startswith(base_prompt):
                    return prompt[len(base_prompt):].strip()
            except ImportError:
                pass
            match = re.search(r'then provide your empathetic response\.\s*(.*)$', prompt, re.DOTALL)
            if match:
                return match.group(1).strip()
            return prompt

        if dataset_name == 'URFunny':
            match = re.search(r'Here is what the speaker says:\s*"(.*)"\s*\.?$', prompt)
            if match:
                return match.group(1).strip()
            return prompt

        if dataset_name in {'MIntRec', 'MIntRec2'}:
            match = re.search(r"Here's the character says in the video:\s*(.*)\s*\.?$", prompt, re.DOTALL)
            if match:
                return match.group(1).strip()
            return prompt

        if dataset_name == 'Mustard':
            memory_match = re.search(r'<memory>(.*?)</memory>', prompt, re.DOTALL)
            speaker_match = re.search(r'Now the speaker,\s*(.*?),\s*said:\s*"(.*)"\s*Answer yes or no\.?$', prompt, re.DOTALL)

            parts = []
            if memory_match:
                history = memory_match.group(1).strip()
                if history:
                    parts.append(f"Chat history:\n{history}")

            if speaker_match:
                speaker = speaker_match.group(1).strip()
                utterance = speaker_match.group(2).strip()
                parts.append(f'Current speaker: {speaker}\nCurrent utterance: "{utterance}"')

            if parts:
                return "\n\n".join(parts)
            return prompt

        return prompt

    def load_sample_names(self, dataset_name: str) -> List[str]:
        print(f"Loading sample names for {dataset_name}...")
        label_path = PATH_TO_LABEL.get(dataset_name)
        if not label_path or not os.path.exists(label_path):
            print(f"Warning: Label file not found for {dataset_name}: {label_path}")
            return []

        sample_names = []

        if label_path.endswith('.csv'):
            with open(label_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'name' in row:
                        sample_names.append(row['name'])
        elif label_path.endswith('.npz'):
            data = np.load(label_path, allow_pickle=True)
            for key in data.files:
                corpus = data[key].item()
                if isinstance(corpus, dict):
                    sample_names.extend(list(corpus.keys()))
            sample_names = list(set(sample_names))

        print(f"  Found {len(sample_names)} samples")
        return sample_names

    def load_transcriptions(self, dataset_name: str) -> Dict[str, str]:
        print(f"Loading transcriptions for {dataset_name}...")
        trans_path = PATH_TO_TRANSCRIPTIONS.get(dataset_name)
        if not trans_path or not os.path.exists(trans_path):
            print(f"Warning: Transcription file not found for {dataset_name}: {trans_path}")
            return {}

        transcriptions = {}
        with open(trans_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'name' in row:
                    name = row['name']
                    text = ''
                    if 'english' in row and row['english']:
                        text = row['english']
                    elif 'chinese' in row and row['chinese']:
                        text = row['chinese']
                    transcriptions[name] = text
        print(f"  Loaded {len(transcriptions)} transcriptions")
        return transcriptions

    def _build_video_cache(self, video_dir: str):
        if video_dir in self._video_cache:
            return

        print(f"Building video cache for {video_dir}...")
        video_map = {}

        for root, _, files in os.walk(video_dir):
            for file in files:
                name, ext = os.path.splitext(file)
                if ext.lower() in VIDEO_EXTENSIONS:
                    video_map[name] = os.path.join(root, file)

        self._video_cache[video_dir] = video_map
        print(f"  Cached {len(video_map)} videos")

    def find_video_path(self, video_dir: str, sample_name: str) -> str:
        self._build_video_cache(video_dir)
        return self._video_cache[video_dir].get(sample_name)

    def get_samples(self, dataset_name: str) -> Dict[str, Dict]:
        print(f"\nProcessing dataset: {dataset_name}")

        if dataset_name in TESTSET_JSON:
            return self.get_samples_from_json(dataset_name)

        sample_names = self.load_sample_names(dataset_name)
        if not sample_names:
            return {}

        transcriptions = self.load_transcriptions(dataset_name)
        video_dir = PATH_TO_RAW_VIDEO.get(dataset_name)

        samples = {}
        missing_count = 0

        print(f"Finding video files...")
        for name in tqdm(sample_names, desc="Matching videos"):
            video_path = self.find_video_path(video_dir, name)
            if video_path:
                samples[name] = {
                    'video_path': video_path,
                    'transcription': transcriptions.get(name, '')
                }
            else:
                missing_count += 1

        print(f"  Found {len(samples)} videos, missing {missing_count}")
        return samples

    def get_samples_from_json(self, dataset_name: str) -> Dict[str, Dict]:
        json_path = TESTSET_JSON.get(dataset_name)
        if not json_path or not os.path.exists(json_path):
            print(f"Warning: Testset JSON not found for {dataset_name}: {json_path}")
            return {}

        print(f"Loading samples from JSON for {dataset_name}: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples = {}
        for item in data:
            messages = item.get('messages', [])
            videos = item.get('videos', [])
            post_id = item.get('post_id')

            if dataset_name == 'Openr1psy':
                prompt = None

                for msg in messages:
                    if msg.get('role') == 'user':
                        prompt = msg.get('content', '')
                        break

                if prompt and post_id is not None:
                    sample_name = str(post_id)
                    samples[sample_name] = {
                        'video_path': None,
                        'transcription': self._parse_json_prompt_as_transcription(dataset_name, prompt)
                    }
                continue

            if not videos:
                continue

            video_path = videos[0]
            prompt = None

            for msg in messages:
                if msg.get('role') == 'user':
                    prompt = msg.get('content', '')
                    break

            if prompt and os.path.exists(video_path):
                sample_name = os.path.splitext(os.path.basename(video_path))[0]
                samples[sample_name] = {
                    'video_path': video_path,
                    'transcription': self._parse_json_prompt_as_transcription(dataset_name, prompt)
                }

        print(f"  Loaded {len(samples)} samples from JSON")
        return samples

    def get_all_datasets(self) -> Dict[str, Dict[str, Dict]]:
        all_datasets = {}
        for name in self.dataset_names:
            all_datasets[name] = self.get_samples(name)
        return all_datasets
