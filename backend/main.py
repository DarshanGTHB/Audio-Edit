import wave
import numpy as np
from flask import Flask, request, jsonify, send_file
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def clip_and_cast(audio_data):
    return np.clip(audio_data, -32768, 32767).astype(np.int16)


def apply_echo(audio_data, delay, decay, framerate):
    delay_samples = int(delay * framerate)
    echo_audio = np.zeros(len(audio_data) + delay_samples, dtype=np.float64)
    echo_audio[:len(audio_data)] = audio_data

    for i in range(len(audio_data)):
        echo_audio[i + delay_samples] += audio_data[i] * decay

    return clip_and_cast(echo_audio)


def apply_reverb(audio_data, decay, iterations, framerate):
    reverb_audio = audio_data.astype(np.float64)
    for i in range(1, iterations + 1):
        delay_samples = int(framerate * 0.01 * i)
        reverb_audio = np.pad(reverb_audio, (delay_samples, 0), mode='constant')
        reverb_audio[:len(audio_data)] += (audio_data * (decay ** i))

    return clip_and_cast(reverb_audio[:len(audio_data)])


def trim_audio(audio_data, start_time, end_time, framerate):
    start_frame = int(start_time * framerate)
    end_frame = int(end_time * framerate) if end_time > 0 else len(audio_data)
    return audio_data[start_frame:end_frame]


def change_playback_speed(audio_data, speed_factor, framerate):
    indices = np.round(np.arange(0, len(audio_data), 1 / speed_factor)).astype(int)
    indices = indices[indices < len(audio_data)]  
    return audio_data[indices]

def change_playback_speed(audio_data, speed_factor, framerate):

    from scipy.signal import resample
    num_samples = int(len(audio_data) / speed_factor)
    resampled_audio = resample(audio_data, num_samples).astype(np.int16)
    return resampled_audio



@app.route('/apply-echo', methods=['POST'])
def apply_echo_route():
    file = request.files.get('file')
    delay = float(request.form.get('delay', 0.5))
    decay = float(request.form.get('decay', 0.5))

    if not file:
        return jsonify({"error": "No audio file provided"}), 400

    try:
        with wave.open(file, 'rb') as wav_file:
            params = wav_file.getparams()
            framerate = params.framerate
            n_channels = params.nchannels
            sampwidth = params.sampwidth
            audio_data = np.frombuffer(wav_file.readframes(params.nframes), dtype=np.int16)

        processed_audio = apply_echo(audio_data, delay, decay, framerate)

        output = BytesIO()
        with wave.open(output, 'wb') as out_file:
            out_file.setnchannels(n_channels)
            out_file.setsampwidth(sampwidth)
            out_file.setframerate(framerate)
            out_file.writeframes(processed_audio.tobytes())

        output.seek(0)
        return send_file(output, mimetype="audio/wav", as_attachment=True, download_name="echo_audio.wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/apply-reverb', methods=['POST'])
def apply_reverb_route():
    file = request.files.get('file')
    decay = float(request.form.get('decay', 0.5))
    iterations = int(request.form.get('iterations', 5))

    if not file:
        return jsonify({"error": "No audio file provided"}), 400

    try:
        with wave.open(file, 'rb') as wav_file:
            params = wav_file.getparams()
            framerate = params.framerate
            n_channels = params.nchannels
            sampwidth = params.sampwidth
            audio_data = np.frombuffer(wav_file.readframes(params.nframes), dtype=np.int16)

        processed_audio = apply_reverb(audio_data, decay, iterations, framerate)

        output = BytesIO()
        with wave.open(output, 'wb') as out_file:
            out_file.setnchannels(n_channels)
            out_file.setsampwidth(sampwidth)
            out_file.setframerate(framerate)
            out_file.writeframes(processed_audio.tobytes())

        output.seek(0)
        return send_file(output, mimetype="audio/wav", as_attachment=True, download_name="reverb_audio.wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/trim-audio', methods=['POST'])
def trim_audio_route():
    file = request.files.get('file')
    start_time = float(request.form.get('start_time', 0))
    end_time = float(request.form.get('end_time', 0))

    if not file:
        return jsonify({"error": "No audio file provided"}), 400

    try:
        with wave.open(file, 'rb') as wav_file:
            params = wav_file.getparams()
            framerate = params.framerate
            n_channels = params.nchannels
            sampwidth = params.sampwidth
            audio_data = np.frombuffer(wav_file.readframes(params.nframes), dtype=np.int16)

        processed_audio = trim_audio(audio_data, start_time, end_time, framerate)

        output = BytesIO()
        with wave.open(output, 'wb') as out_file:
            out_file.setnchannels(n_channels)
            out_file.setsampwidth(sampwidth)
            out_file.setframerate(framerate)
            out_file.writeframes(processed_audio.tobytes())

        output.seek(0)
        return send_file(output, mimetype="audio/wav", as_attachment=True, download_name="trimmed_audio.wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/change-playback-speed', methods=['POST'])
def change_playback_speed_route():
    file = request.files.get('file')
    speed_factor = float(request.form.get('speed_factor', 1.0))

    if not file:
        return jsonify({"error": "No audio file provided"}), 400

    try:
        with wave.open(file, 'rb') as wav_file:
            params = wav_file.getparams()
            framerate = params.framerate
            n_channels = params.nchannels
            sampwidth = params.sampwidth
            audio_data = np.frombuffer(wav_file.readframes(params.nframes), dtype=np.int16)

        processed_audio = change_playback_speed(audio_data, speed_factor, framerate)

        output = BytesIO()
        with wave.open(output, 'wb') as out_file:
            out_file.setnchannels(n_channels)
            out_file.setsampwidth(sampwidth)
            out_file.setframerate(int(framerate * speed_factor))
            out_file.writeframes(processed_audio.tobytes())

        output.seek(0)
        return send_file(output, mimetype="audio/wav", as_attachment=True, download_name="speed_changed_audio.wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
