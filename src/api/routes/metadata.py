"""
元数据路由

获取说话人列表、语言列表、克隆音色列表等元数据
"""

from fastapi import APIRouter, HTTPException, Depends
from api.models import SpeakersResponse, LanguagesResponse, ClonesResponse, DesignPresetsResponse
from api.dependencies import get_voice_library

router = APIRouter()


# 说话人列表（来自 VoiceLibrary）
CUSTOM_VOICE_SPEAKERS = [
    "Vivian",      # 明亮略带个性的年轻女性
    "Serena",      # 温柔温柔的年轻女性
    "Uncle_Fu",    # 沉稳的中年男性
    "Dylan",       # 清晰自然的北京男性
    "Eric",        # 活泼的成都男性
    "Ryan",        # 动感节奏强的英语男性
    "Aiden",       # 阳光美国男性
    "Ono_Anna",    # 活泼日本女性
    "Sohee"        # 温暖韩国女性
]

# 支持的语言列表
SUPPORTED_LANGUAGES = {
    "Chinese": "中文",
    "English": "英语",
    "Japanese": "日语",
    "Korean": "韩语",
    "Auto": "自动检测"
}

# 语音设计内置预设
VOICE_DESIGN_PRESETS = {
    "温柔女声": "体现温柔细腻的女性声音，音调柔和，语速舒缓，营造出温暖、亲切的听觉效果。",
    "活泼少女": "体现活泼可爱的少女声音，音调偏高且富有弹性，语速轻快，营造出年轻、活力四射的听觉效果。",
    "磁性大叔": "体现磁性深沉的男性声音，音调偏低且富有质感，语速稳重，营造出成熟、有魅力的听觉效果。",
    "正太少年": "体现清脆自然的少年声音，音调适中，语速流畅，营造出纯真、少年感的听觉效果。",
    "知性御姐": "体现成熟知性的女性声音，音调中低，语速从容，营造出优雅、有韵味的听觉效果。",
    "沉稳长者": "体现稳重沧桑的长者声音，音调低沉，语速缓慢有力，营造出阅历丰富、值得信赖的听觉效果。",
}


@router.get("/speakers", response_model=SpeakersResponse)
async def get_speakers():
    """
    获取支持的说话人列表

    返回 Custom Voice 模式可用的预设说话人
    """
    return {
        "success": True,
        "speakers": CUSTOM_VOICE_SPEAKERS
    }


@router.get("/languages", response_model=LanguagesResponse)
async def get_languages():
    """
    获取支持的语言列表

    返回所有支持的语言及其显示名称
    """
    return {
        "success": True,
        "languages": SUPPORTED_LANGUAGES
    }


@router.get("/clones", response_model=ClonesResponse)
async def get_clones(voice_library=Depends(get_voice_library)):
    """
    获取保存的克隆音色列表

    返回用户保存的所有语音克隆音色
    """
    if voice_library is None:
        return {
            "success": True,
            "clones": []
        }

    clones = voice_library.get_all_clones()
    return {
        "success": True,
        "clones": clones
    }


@router.get("/design-presets", response_model=DesignPresetsResponse)
async def get_design_presets(voice_library=Depends(get_voice_library)):
    """
    获取语音设计预设列表

    返回内置和用户自定义的语音设计预设
    """
    if voice_library is None:
        # 如果 VoiceLibrary 不可用，返回内置预设
        return {
            "success": True,
            "presets": VOICE_DESIGN_PRESETS
        }

    # 获取所有预设（内置 + 用户自定义）
    presets = voice_library.get_all_design_presets()
    return {
        "success": True,
        "presets": presets
    }
