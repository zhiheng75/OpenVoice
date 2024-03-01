# The code is modified from the original OpenVoice implementation.

import torch
import se_extractor
from api import BaseSpeakerTTS, ToneColorConverter
import argparse
import logging
from typing import Dict

LANG_MAP = {
    "zh": "Chinese",
    "en": "English",
}


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--text",
        type=str,
        default="今天天气真好，我们一起出去吃饭吧。",
        help="text to be synthesized",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="zh",
        choices=['zh', 'en'],
        help="language of the text",
    )
    parser.add_argument(
        "--ckpt-base",
        type=str,
        default="checkpoints/base_speakers",
        help="base speaker model ckpt directory",
    )
    parser.add_argument(
        "--ckpt-converter",
        type=str,
        default="checkpoints/converter",
        help="tone color converter model ckpt directory",
    )
    parser.add_argument(
        "--source-style",
        type=str,
        default="zh_default_se.pth",
        help="tone color converter model ckpt directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="directory to save the output audio",
    )
    parser.add_argument(
        "--save-filename",
        type=str,
        default="output_whispering.mp3",
        help="output target speaker audio file",
    )
    parser.add_argument(
        "--reference-speaker",
        type=str,
        default="resources/example_reference.mp3",
        help="reference speaker audio file",
    )

    return parser


def load_tts_models(ckpt_base: str, device: str) -> Dict:
    model_dict = {}
    for lang in LANG_MAP.keys():
        model_dict[lang] = BaseSpeakerTTS(
            f'{ckpt_base}/{lang.upper()}/config.json', device=device)
        model_dict[lang].load_ckpt(f'{ckpt_base}/{lang.upper()}/checkpoint.pth')
    return model_dict


def load_converter_model(ckpt_converter: str,
                         device: str) -> ToneColorConverter:
    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json',
                                              device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
    return tone_color_converter


def gen_basespeaker_tts(text: str,
                        tts_model: BaseSpeakerTTS,
                        output_path: str,
                        speaker: str = 'default',
                        language: str = 'English',
                        speed: float = 1.0) -> None:
    tts_model.tts(text, output_path, speaker, language, speed)


def tone_convert(src_path: str, tone_color_converter: ToneColorConverter,
                 src_se: str, reference_speaker: str, save_path: str,
                 device: str, output_dir: str) -> None:
    logging.info(f"Converting {src_path} to {save_path}")
    source_se = torch.load(src_se).to(device)
    # source_se, _ = se_extractor.get_se(src_path,
    #                                    tone_color_converter,
    #                                    target_dir=f'{output_dir}/processed',
    #                                    vad=True)
    target_se, _ = se_extractor.get_se(reference_speaker,
                                       tone_color_converter,
                                       target_dir=f'{output_dir}/processed',
                                       vad=True)
    tone_color_converter.convert(audio_src_path=src_path,
                                 src_se=source_se,
                                 tgt_se=target_se,
                                 output_path=save_path,
                                 message="@MyShell")


def main():
    parser = get_parser()
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_dict = load_tts_models(args.ckpt_base, device)
    tone_converter = load_converter_model(args.ckpt_converter, device)
    base_speaker_output_path = f"{args.output_dir}/base_speaker_tmp.wav"
    gen_basespeaker_tts(text=args.text,
                        tts_model=model_dict[args.lang],
                        output_path=base_speaker_output_path,
                        language=LANG_MAP[args.lang])
    tone_convert(
        src_path=base_speaker_output_path,
        tone_color_converter=tone_converter,
        src_se=f'{args.ckpt_base}/{args.lang.upper()}/{args.source_style}',
        reference_speaker=args.reference_speaker,
        save_path=f'{args.output_dir}/{args.save_filename}',
        device=device,
        output_dir=args.output_dir)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
