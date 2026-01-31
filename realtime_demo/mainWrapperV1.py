"""
Instructions for the Student
Baseline Test: Set CURRENT_CODEC = "PCM" which "perfect" reference data. (Just for comparison)
Verify Custom Logic: Set CURRENT_CODEC = "CUSTOM". This runs the Dummy Compression (You need to replace it).
Implement Your Algorithm: * Locate the # --- REPLACE START --- blocks inside the CustomCodec class.
Delete the Dummy Logic and paste your initialization, encoding, and decoding code.
Ensure your encode returns compressed bytes and decode returns raw 16-bit PCM bytes.
"""
import socket
import struct
import threading
import time
import csv
import pyaudio
import numpy as np
import wave
# import student_codec # Import the student's file
import std_enc_dec

# ==============================
# 1. CODEC SELECTION
# ==============================
CURRENT_CODEC = "CUSTOM" 

# ==============================
# Configuration Parameters
# ==============================
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_SIZE = 320  # 20ms frames - model handles small frames reasonably
FRAME_DURATION_MS = (FRAME_SIZE / SAMPLE_RATE) * 1000
BITRATE = 8000 
SILENCE_THRESHOLD = 100

###################################
FORMAT = pyaudio.paInt16
RAW_BYTES_PER_FRAME = FRAME_SIZE * 2 
LOCAL_IP = "0.0.0.0"
LOCAL_PORT = 5000
REMOTE_IP = "127.0.0.1"
REMOTE_PORT = 5000

# --- NEW: Storage for post-processing metrics (PESQ/STOI) ---
all_orig_frames = []
all_decoded_frames = []

# ==============================
# Custom Codec Class
# ==============================
class CustomCodec:
    def __init__(self, mode, rate, channels, bitrate):
        self.mode = mode
        self.lookahead = std_enc_dec.LOOKAHEAD_MS if mode == "CUSTOM" else 0.0
        if mode == "CUSTOM":
            print("--- Hooked into std_enc_dec.py ---")
            pass

    def encode(self, pcm_data):
        if self.mode == "CUSTOM":
            audio_frame = np.frombuffer(pcm_data, dtype=np.int16)
            return std_enc_dec.my_encoder_logic(audio_frame)
        return pcm_data 

    def decode(self, codec_data, is_tester=False):
        if self.mode == "CUSTOM":
            audio_frame = std_enc_dec.my_decoder_logic(codec_data)
            
            # Safety check to ensure student returns numpy array
            if not isinstance(audio_frame, np.ndarray):
                audio_frame = np.array(audio_frame)
            
            return audio_frame.tobytes()
        return codec_data
 
# ==============================
# Initialize Codec & Audio
# ==============================
codec = CustomCodec(CURRENT_CODEC, SAMPLE_RATE, CHANNELS, BITRATE)
pa = pyaudio.PyAudio()
stream_in = pa.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, 
                    input=True, frames_per_buffer=FRAME_SIZE)
stream_out = pa.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, 
                     output=True, frames_per_buffer=FRAME_SIZE)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LOCAL_IP, LOCAL_PORT))

# Increase buffer sizes for larger packets
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)

codec_csv = open(f"evaluation_{CURRENT_CODEC}.csv", "w", newline="")
network_csv = open(f"network_{CURRENT_CODEC}.csv", "w", newline="")
codec_writer = csv.writer(codec_csv)
net_writer = csv.writer(network_csv)

codec_writer.writerow(["seq", "encode_ms", "decode_ms_local", "algorithmic_delay_ms", 
                       "bitrate_kbps", "similarity_score", "compression_ratio", "space_saving_pct"])
net_writer.writerow(["seq", "rtt_ms", "one_way_ms", "jitter_ms"])

def calculate_similarity(original, decoded):
    orig = np.frombuffer(original, dtype=np.int16).astype(np.float32)
    dec = np.frombuffer(decoded, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(orig**2))
    if rms < SILENCE_THRESHOLD: 
        return float('nan')
    if np.std(orig) == 0: 
        return 100.0 if np.array_equal(orig, dec) else 0.0
    return max(0, np.corrcoef(orig, dec)[0, 1] * 100)

seq_counter = 0
pending_rtt = {}
last_arrival = None
HEADER_FMT = "!Id"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# ==============================
# Processing Threads
# ==============================
def sender():
    global seq_counter
    print(f"SENDER ACTIVE: {CURRENT_CODEC} Mode")
    while True:
        try:
            t_start = time.time()
            pcm = stream_in.read(FRAME_SIZE, exception_on_overflow=False)
            
            # Measure Encoding Latency
            t0 = time.perf_counter()
            encoded_payload = codec.encode(pcm)
            enc_time = (time.perf_counter() - t0) * 1000
            
            # Local Decode for Quality Assessment
            t1 = time.perf_counter()
            pcm_loopback = codec.decode(encoded_payload, is_tester=True)
            dec_time_local = (time.perf_counter() - t1) * 1000
            
            # --- NEW: Capture Audio for Post-Processing (Outside latency timers) ---
            all_orig_frames.append(pcm)
            all_decoded_frames.append(pcm_loopback)
            
            # Metrics Calculations
            sim_score = calculate_similarity(pcm, pcm_loopback)
            comp_size = len(encoded_payload)
            comp_ratio = RAW_BYTES_PER_FRAME / comp_size
            space_saving = (1 - (comp_size / RAW_BYTES_PER_FRAME)) * 100
            bitrate_actual = (comp_size * 8) / (FRAME_DURATION_MS / 1000) / 1000
            alg_delay = FRAME_DURATION_MS + codec.lookahead
            
            header = struct.pack(HEADER_FMT, seq_counter, t_start)
            sock.sendto(header + encoded_payload, (REMOTE_IP, REMOTE_PORT))
            pending_rtt[seq_counter] = t_start
            
            codec_writer.writerow([seq_counter, f"{enc_time:.4f}", f"{dec_time_local:.4f}", 
                                   f"{alg_delay:.1f}", f"{bitrate_actual:.2f}", 
                                   f"{sim_score:.2f}" if not np.isnan(sim_score) else "NaN",
                                   f"{comp_ratio:.2f}", f"{space_saving:.2f}"])
            seq_counter += 1
        except Exception as e: 
            print(f"Sender Error: {e}")

def receiver():
    global last_arrival
    while True:
        try:
            packet, addr = sock.recvfrom(65536)  # Increased from 4096
            recv_time = time.time()
            
            if len(packet) == HEADER_SIZE:
                s_seq, s_ts = struct.unpack(HEADER_FMT, packet)
                if s_seq in pending_rtt:
                    rtt = (recv_time - pending_rtt.pop(s_seq)) * 1000
                    jitter = abs((recv_time - last_arrival) * 1000 - FRAME_DURATION_MS) if last_arrival else 0
                    last_arrival = recv_time
                    net_writer.writerow([s_seq, f"{rtt:.2f}", f"{rtt/2:.2f}", f"{jitter:.2f}"])
            elif len(packet) > HEADER_SIZE:
                sock.sendto(packet[:HEADER_SIZE], addr) 
                payload = packet[HEADER_SIZE:]
                decoded_audio = codec.decode(payload)
                stream_out.write(decoded_audio)
        except OSError as e:
            if e.errno == 9:  # Bad file descriptor - socket closed during shutdown
                break
            print(f"Receiver Error: {e}")
        except Exception as e: 
            print(f"Receiver Error: {e}")

threading.Thread(target=receiver, daemon=True).start()
threading.Thread(target=sender, daemon=True).start()

# ==============================
# Shutdown & WAV Saving
# ==============================
try:
    while True: 
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down and saving data...")
    
    # Define WAV saving helper
    def save_wav(filename, frames):
        if not frames: 
            return
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pa.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames))
        print(f"Saved: {filename}")
    
    # Save audio files for PESQ/STOI analysis
    save_wav(f"original_{CURRENT_CODEC}.wav", all_orig_frames)
    save_wav(f"decoded_{CURRENT_CODEC}.wav", all_decoded_frames)
    codec_csv.close()
    network_csv.close()
    sock.close()
    stream_in.stop_stream()
    stream_out.stop_stream()
    pa.terminate()
    print("All CSVs and Audio files saved successfully.")
