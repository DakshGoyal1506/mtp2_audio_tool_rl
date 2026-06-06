# #!/usr/bin/env python3
# """
# Test script to verify audio processing tools are working correctly.
# Run this after fixing the huggingface_hub version issue.
# """

# import json
# import asyncio
# import os
# from pathlib import Path
# from datetime import datetime
# from tqdm import tqdm
# import traceback
# import sys

# # Add parent directory to path for imports
# sys.path.insert(0, str(Path(__file__).parent))

# from audio_maestro.audio_copilot import AudioCopilotTool

# # Test audio file - use the first one from test-mini-audios
# TEST_AUDIO = "./test-mini-audios/3fe64f3d-282c-4bc8-a753-68f8f6c35652.wav"


# def print_result(name: str, result: dict):
#     """Pretty print test result"""
#     if "error" in result:
#         print(f"  ❌ {name}: FAILED - {result['error']}")
#         return False
#     else:
#         print(f"  ✅ {name}: PASSED")
#         # Print a brief summary of the result
#         for key, value in list(result.items())[:3]:
#             if key != "function":
#                 print(f"      {key}: {str(value)[:80]}...")
#         return True


# async def test_all_tools():
#     """Test all audio processing tools"""

#     if not os.path.exists(TEST_AUDIO):
#         print(f"❌ Test audio file not found: {TEST_AUDIO}")
#         print("Please ensure the test audio files are available.")
#         return

#     print(f"\n🎵 Testing audio tools with: {TEST_AUDIO}\n")
#     print("=" * 60)

#     tool = AudioCopilotTool()
#     results = {}

#     # List of tools to test - these are the ones that were failing
#     tools_to_test = [
#         ("speaker_diarization", lambda: tool.speaker_diarization(TEST_AUDIO)),
#         ("emotion_recognition", lambda: tool.emotion_recognition(TEST_AUDIO)),
#         ("speech_recognition", lambda: tool.speech_recognition(TEST_AUDIO)),
#         ("stressed_analysis", lambda: tool.stressed_analysis(TEST_AUDIO)),
#         ("sound_classification", lambda: tool.sound_classification(TEST_AUDIO)),
#         ("sound_duration_analysis", lambda: tool.sound_duration_analysis(TEST_AUDIO)),
#         ("genre_analysis", lambda: tool.genre_analysis(TEST_AUDIO)),
#         ("speech_to_noise_ratio", lambda: tool.speech_to_noise_ratio(TEST_AUDIO)),
#         ('get_audio_features', lambda: tool.get_audio_features(TEST_AUDIO)),
#     ]

#     passed = 0
#     failed = 0

#     for name, func in tools_to_test:
#         print(f"\n📋 Testing: {name}")
#         try:
#             result = await func()
#             results[name] = result
#             if print_result(name, result):
#                 passed += 1
#             else:
#                 failed += 1
#         except Exception as e:
#             print(f"  ❌ {name}: EXCEPTION - {str(e)}")
#             results[name] = {"error": str(e)}
#             failed += 1

#     print("\n" + "=" * 60)
#     print(f"\n📊 Summary: {passed} passed, {failed} failed out of {len(tools_to_test)} tests\n")

#     # Specifically check for the hf_hub_download error
#     hf_error_found = False
#     for name, result in results.items():
#         if "error" in result and "use_auth_token" in str(result.get("error", "")):
#             hf_error_found = True
#             print(f"⚠️  '{name}' still has the huggingface_hub version issue!")

#     if hf_error_found:
#         print("\n💡 To fix the huggingface_hub issue, run one of:")
#         print('   pip install "huggingface_hub<0.24"')
#         print("   OR")
#         print("   pip install --upgrade pyannote-audio")
#     elif failed == 0:
#         print("🎉 All tools are working correctly!")

#     return results


# if __name__ == "__main__":
#     print("\n🔧 Audio Maestro - Tool Verification Script")
#     print("=" * 60)

#     # Check huggingface_hub version
#     try:
#         import huggingface_hub
#         print(f"huggingface_hub version: {huggingface_hub.__version__}")
#     except ImportError:
#         print("huggingface_hub not installed!")

#     # Check pyannote version
#     try:
#         import pyannote.audio
#         print(f"pyannote.audio version: {pyannote.audio.__version__}")
#     except ImportError:
#         print("pyannote.audio not installed!")
#     except AttributeError:
#         print("pyannote.audio installed (version unknown)")

#     # Run tests
#     asyncio.run(test_all_tools())


import autochord