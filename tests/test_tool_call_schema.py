import json

from mtp2_audio_tool_rl.tool_calling.validate_tool_call import validate_tool_call


def valid_call():
    return {
        "tool_name": "speaker_diarization",
        "arguments": {"audio_path": "audio/sample_001.wav"},
    }


def test_valid_dict_tool_call_passes():
    result = validate_tool_call(valid_call())

    assert result.valid
    assert result.error_code is None
    assert result.tool_call.tool_name == "speaker_diarization"


def test_valid_json_string_passes():
    result = validate_tool_call(json.dumps(valid_call()))

    assert result.valid
    assert result.tool_call.arguments["audio_path"] == "audio/sample_001.wav"


def test_malformed_json_fails():
    result = validate_tool_call("{not valid json")

    assert not result.valid
    assert result.error_code == "malformed_json"


def test_missing_tool_name_fails():
    data = valid_call()
    del data["tool_name"]

    result = validate_tool_call(data)

    assert not result.valid
    assert result.error_code == "missing_tool_name"


def test_unknown_tool_fails():
    data = valid_call()
    data["tool_name"] = "spectrogram_plotter"

    result = validate_tool_call(data)

    assert not result.valid
    assert result.error_code == "unknown_tool"


def test_missing_arguments_fails():
    data = valid_call()
    del data["arguments"]

    result = validate_tool_call(data)

    assert not result.valid
    assert result.error_code == "missing_arguments"


def test_non_dict_arguments_fails():
    data = valid_call()
    data["arguments"] = ["audio/sample_001.wav"]

    result = validate_tool_call(data)

    assert not result.valid
    assert result.error_code == "invalid_arguments_type"


def test_absolute_audio_path_fails():
    data = valid_call()
    data["arguments"]["audio_path"] = "/data/sample_001.wav"

    result = validate_tool_call(data)

    assert not result.valid
    assert result.error_code == "unsafe_path"


def test_parent_traversal_path_fails():
    data = valid_call()
    data["arguments"]["audio_path"] = "../sample_001.wav"

    result = validate_tool_call(data)

    assert not result.valid
    assert result.error_code == "unsafe_path"
