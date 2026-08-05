import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据集集合配置
DATASET_COLLECTIONS = {
    'merunibench': [
        'OVMERD',
        'MER2023',
        'MER2024',
        'IEMOCAPFour',
        'CMUMOSI',
        'CMUMOSEI',
        'SIMS',
        'SIMSv2',
        'MELD'
    ],
    'mir': [
        'MIntRec',
        'MIntRec2',
    ],
    'msd': [
        'Mustard'
    ],
    'mhd': [
        'URFunny'
    ],
    'erg': [
        'AvaMERG'
    ],
    'esc': [
        'Openr1psy'
    ]
}

DATA_DIR = {
    'MER2025OV':      '/path/to/your/resource/MER2025',
    'MERCaptionPlus': '/path/to/your/resource/MER2025',
    'OVMERD': '/path/to/your/resource/MER2025',
    'MER2023': '/path/to/your/resource/MER-UniBench/MER-UniBench/mer2023-dataset-process',
    'MER2024': '/path/to/your/resource/MER-UniBench/MER-UniBench/mer2024-dataset-process',
    'IEMOCAPFour': '/path/to/your/resource/MER-UniBench/MER-UniBench/iemocap-process',
    'CMUMOSI': '/path/to/your/resource/MER-UniBench/MER-UniBench/cmumosi-process',
    'CMUMOSEI': '/path/to/your/resource/MER-UniBench/MER-UniBench/cmumosei-process',
    'SIMS': '/path/to/your/resource/MER-UniBench/MER-UniBench/sims-process',
    'SIMSv2': '/path/to/your/resource/MER-UniBench/MER-UniBench/simsv2-process',
    'MELD': '/path/to/your/resource/MER-UniBench/MER-UniBench/meld-process',
    'AvaMERG':        '/path/to/your/resource/AvaMERG',
    'MELD_train':     '/path/to/your/resource/MMAFFIn/videos/',
    'IEMOCAP_train':  '/path/to/your/resource/IEMOCAP-R/',
    'MERRFine':       '/path/to/your/resource/MER2023',
    'EMER':           '/path/to/your/resource/MER2025',
    'CAER_FERV39K':   '/path/to/your/resource/CAER_FERV39K',
    'M3ED':           '/path/to/your/resource/M3ED',
    'CREMA':          '/path/to/your/resource/CREMA-D',
    'MIntRec':        '/path/to/your/resource/MIntRec',
    'MIntRec2':       '/path/to/your/resource/MIntRec2',
    'OpenR1Psy':      '/path/to/your/resource/OpenR1Psy',
    'URFunny':        '/path/to/your/resource/URFunny',
    'Mustard':        '/path/to/your/resource/Mustard' 
}

PATH_TO_RAW_VIDEO = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'video','video'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'video','video'),
    'OVMERD': os.path.join(DATA_DIR['OVMERD'], 'video', 'video'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'video'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'subvideo-tgt'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'subvideo'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'subvideo_new'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'video'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'subvideo'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'video_new'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'video'),
    'AvaMERG': os.path.join(DATA_DIR['AvaMERG'], 'video'),
    'MELD_train': os.path.join(DATA_DIR['MELD_train'], 'MELD'),
    'IEMOCAP_train': os.path.join(DATA_DIR['IEMOCAP_train'], 'video'),
    'MERRFine': os.path.join(DATA_DIR['MERRFine'], 'test3'),
    'EMER':  os.path.join(DATA_DIR['EMER'], 'video','video'),
    'CAER_FERV39K': os.path.join(DATA_DIR['CAER_FERV39K'], 'video'),
    'MIntRec': os.path.join(DATA_DIR['MIntRec'], 'video'),
    'MIntRec2': os.path.join(DATA_DIR['MIntRec2'], 'video'),
    'URFunny': os.path.join(DATA_DIR['URFunny'],'videos'),
}

PATH_TO_LABEL = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'mer2025-ov.csv'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'track3_train_mercaptionplus.csv'),
    'OVMERD': os.path.join(DATA_DIR['OVMERD'], 'track2_train_ovmerd.csv'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'label-6way.npz'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'label_4way.npz'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'label.npz'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'label.npz'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'label.npz'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'label.npz'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'label.npz'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'label-6way.npz'),
    'AvaMERG': os.path.join(DATA_DIR['AvaMERG'], 'train.json'),# v_test_v5_0.json是有完整caption的测试标签
    'MELD_train': os.path.join(DATA_DIR['MELD_train'], 'MELD_train.json'),
    'IEMOCAP_train': os.path.join(DATA_DIR['IEMOCAP_train'], ''),
    'MERRFine': os.path.join(DATA_DIR['MERRFine'], 'MERR_fine_grained.json'),
    'OpenR1Psy': os.path.join(DATA_DIR['OpenR1Psy'], 'train.json'), #训练集
}

PATH_TO_TRANSCRIPTIONS = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'subtitle_chieng.csv'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'subtitle_chieng.csv'),
    'OVMERD': os.path.join(DATA_DIR['OVMERD'], 'subtitle_chieng.csv'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'transcription-engchi-polish.csv'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'transcription-engchi-polish.csv'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'transcription-engchi-polish.csv'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'transcription-engchi-polish.csv'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'transcription-engchi-polish.csv'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'transcription-engchi-polish.csv'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'transcription-engchi-polish.csv'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'transcription_merge.csv'),
}

TESTSET_JSON = {
    'MIntRec': os.path.join(BASE_DIR, 'datas/eval/testset_mintrec.json'),
    'MIntRec2': os.path.join(BASE_DIR, 'datas/eval/testset_mintrec2.json'),
    'URFunny': os.path.join(BASE_DIR, 'datas/eval/testset_urfunny.json'),
    'Mustard': os.path.join(BASE_DIR, 'datas/eval/testset_mustard.json'),
    'AvaMERG': os.path.join(BASE_DIR, 'datas/eval/testset_avamerg.json'),
    'Openr1psy': os.path.join(BASE_DIR, 'datas/eval/test_openr1psy.json')
}

VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

MODEL_PATH = '/path/to/your/resource/Qwen3.5-4B'
MODEL_NAME = os.path.basename(MODEL_PATH)
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

CHECKPOINT_NAME = 'checkpoint_000000_loss_0.000.npz'
