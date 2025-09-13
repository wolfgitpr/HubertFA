import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio


def compute_power_curve(waveform, sample_rate, hop_size, target_frames, device='cpu'):
    audio_length = waveform.size(-1)
    window_size = int(0.02 * sample_rate)

    if window_size % 2 == 0:
        window_size += 1

    half_window = window_size // 2

    frame_centers = torch.arange(0, target_frames, device=device) * hop_size + hop_size // 2

    window_starts = (frame_centers - half_window).clamp(min=0)

    indices = torch.arange(window_size, device=device).unsqueeze(0)  # [1, window_size]
    window_indices = window_starts.unsqueeze(1) + indices  # [target_frames, window_size]

    window_indices = torch.clamp(window_indices, 0, audio_length - 1)
    window_audio = waveform[window_indices.long()]  # [target_frames, window_size]

    valid_mask = (window_indices >= 0) & (window_indices < audio_length)
    squared_sum = (window_audio ** 2).sum(dim=1)
    valid_count = valid_mask.sum(dim=1).float()

    rms = torch.sqrt(squared_sum / valid_count.clamp(min=1))

    return 20 * torch.log10(rms.clamp(min=1e-10) / 1.0)  # power_db [T]


def load_and_process_audio(file_path, device='cpu'):
    waveform, sample_rate = torchaudio.load(file_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform = waveform.to(device)

    hop_duration = 0.01
    hop_size = int(hop_duration * sample_rate)

    audio_length = waveform.size(-1)
    target_frames = int(np.ceil(audio_length / hop_size))

    power_db = compute_power_curve(
        waveform.squeeze(0), sample_rate, hop_size, target_frames, device
    )

    return waveform, sample_rate, power_db, hop_size


def plot_waveform_and_power(waveform, sample_rate, power_db, hop_size):
    time_axis_waveform = torch.arange(waveform.size(1)) / sample_rate
    time_axis_power = (torch.arange(power_db.size(0)) * hop_size + hop_size // 2) / sample_rate

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(time_axis_waveform.numpy(), waveform.squeeze().numpy())
    ax1.set_title('Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)

    ax2.plot(time_axis_power.numpy(), power_db.numpy())
    ax2.set_title('Power Curve (dB)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Power (dB)')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    wav_file_path = r"C:\Users\99662\Desktop\hfa测试wav\wav\凑热闹_000.wav"

    waveform, sample_rate, power_db, hop_size = load_and_process_audio(wav_file_path)

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Audio duration: {waveform.size(1) / sample_rate:.2f} seconds")
    print(f"Number of power curve points: {len(power_db)}")

    plot_waveform_and_power(waveform, sample_rate, power_db, hop_size)
