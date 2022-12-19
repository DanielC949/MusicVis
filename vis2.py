import math
import sys, subprocess, bisect, time
import pygame
from pygame.locals import *
import tkinter as tk
from tkinter import N, filedialog
from scipy.io import wavfile
from scipy import signal
from scipy import ndimage
import librosa
import numpy
import numpy.typing

from typing import Iterable, Iterator, List, Tuple

WHITE = (0xff, 0xff, 0xff)
BLACK = (0, 0, 0)
RED = (0xff, 0, 0)
GREEN = (0, 0xff, 0)
BLUE = (0, 0, 0xff)
YELLOW = (0xff, 0xff, 0)
CYAN = (0, 0xff, 0xff)
MAGENTA = (0xff, 0, 0xff)
GRAY = (0xaa, 0xaa, 0xaa)
LIGHT_GRAY = (0xcc, 0xcc, 0xcc)
LIGHT_BLUE = (0x44, 0x44, 0xff)
def color_interpolate(val: float, min: float, max: float, colors: List[Tuple[int, int, int]]=[GREEN, YELLOW, RED]):
    if val >= max:
        return colors[-1]
    elif val <= min:
        return colors[0]
    
    proportion = (val - min) / (max - min)
    i = int((len(colors) - 1) * proportion)
    low, high = colors[i], colors[i + 1]
    n_prop = (proportion - (i / (len(colors) - 1))) * (len(colors) - 1)
    return tuple((int(low[i] + (high[i] - low[i]) * n_prop)) for i in range(3))

semitone, half_semitone = 2 ** (1/12), 2 ** (1/24)
note_ranges = [(t / half_semitone, t * half_semitone) for t in (27.5 * (semitone ** i) for i in range(88))]
note_labels = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
def freq_to_idx(freq: float) -> int:
    lo, hi = 0, len(note_ranges) - 1

    if freq < note_ranges[lo][0] or freq > note_ranges[hi][1]:
        return -1

    while lo < hi:
        mid = (lo + hi) // 2

        if freq >= note_ranges[mid][0] and freq <= note_ranges[mid][1]:
            return mid
        elif freq <= note_ranges[mid][0]:
            hi = mid - 1
        else:
            lo = mid + 1
    return hi
def note_from_idx(idx: int) -> str:
    return f'{note_labels[idx % 12]}{idx // 12 + (1 if idx % 12 >= 3 else 0)}'
def note_from_freq(freq: float) -> str:
    idx = freq_to_idx(freq)
    return note_from_idx(idx) if idx != -1 else ''

class Spectogram_Wrapper:
    f : numpy.typing.NDArray
    t : numpy.typing.NDArray
    sxx : numpy.typing.NDArray
    samplerate : int
    maxamp : float
    length : float
    lazy: bool

    def __init__(self, filename: str, samples_per_second: int, lazy: bool = True) -> None:
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-i', filename, 'temp/temp.wav'], check=True)
        self.samplerate, audio = wavfile.read('temp/temp.wav')
        self.length = audio.shape[0] / self.samplerate
        self.lazy = lazy
        audio = librosa.to_mono(numpy.transpose(audio.astype(numpy.float64)))
        
        print('raw audio summary:', numpy.min(audio), numpy.max(audio), numpy.average(audio), numpy.std(audio))
        audio = self.filter_raw(audio, samples_per_second)
        print('filtered summary:', numpy.min(audio), numpy.max(audio), numpy.average(audio), numpy.std(audio))
        n_per_frame = self.samplerate // samples_per_second
        overlap_frames = int(0.5 * n_per_frame)
        #self.f, self.t, self.sxx = signal.spectrogram(audio, self.samplerate,
            #mode='magnitude', nperseg=n_per_frame + overlap_frames, noverlap=overlap_frames, scaling='density')
        self.f = librosa.fft_frequencies(sr=self.samplerate, n_fft=n_per_frame)
        S = numpy.abs(librosa.stft(audio, n_fft=n_per_frame, hop_length=n_per_frame))
        self.sxx = librosa.salience(S, freqs=self.f, harmonics=[1], weights=[1], fill_value=0)
        self.t = numpy.linspace(0, self.length, num=self.sxx.shape[1])
        print('comptued spectrogram')
        print('spectrogram summary:', numpy.min(self.sxx), numpy.max(self.sxx), numpy.average(self.sxx), numpy.std(self.sxx))
        #self.filter()
        if not lazy:
            self.bin_spectrogram(self.f, self.sxx)
            self.maxamp = numpy.max(numpy.abs(self.sxx)) * 1.05
        else:
            self.maxamp = 2e6
            #next(self.iterator(self.length / 2))
    
    def filter_percussion(self, audio: numpy.typing.NDArray) -> numpy.typing.NDArray:
        S, phase = librosa.magphase(librosa.core.spectrum.stft(audio))
        kernel = 4
        margin = 2.5

        h_shape = [1 for _ in S.shape]
        h_shape[-1] = kernel
        p_shape = [1 for _ in S.shape]
        p_shape[-2] = kernel

        harm = numpy.empty_like(S)
        harm[:] = ndimage.median_filter(S, size=h_shape, mode='reflect')
        perc = numpy.empty_like(S)
        perc[:] = ndimage.median_filter(S, size=p_shape, mode='reflect')

        mask = librosa.util.softmask(harm, perc * margin, power=2, split_zeros=False)
        res = (S * mask) * phase

        res = librosa.core.istft(res)
        return res

    def filter_raw(self, audio: numpy.typing.NDArray, samples_per_second: int):
        audio = self.filter_percussion(audio)
        sensitivity_floor = samples_per_second / (2**(1/12) - 1)
        butter_ord, butter_wn = signal.buttord([sensitivity_floor, 440 * 4], [sensitivity_floor / 2, 440 * 5], gpass=1, gstop=20, fs=self.samplerate)
        butter = signal.butter(butter_ord, butter_wn, btype='bandpass', output='sos', fs=self.samplerate)
        res = signal.sosfilt(butter, audio)
        return res
    
    def bin_spectrogram(self, freqs, sxx) -> None:
        res = numpy.zeros((sxx.shape[1], len(note_ranges)))
        for i in range(sxx.shape[1]):
            print(f'Analyzing spectrogram, {i / sxx.shape[1] * 100:2.2f}% complete...', end='\r')
            for f, a in zip(freqs, sxx[:, i]):
                idx = freq_to_idx(f)
                if i == -1:
                    continue
                res[i, idx] = max(res[i, idx], a)
        self.sxx = res
        print('\nFinished spectrogram analysis')
    
    def iterator(self, start_time: float = 0) -> Iterator[numpy.typing.NDArray]:
        if start_time < 0 or start_time > self.length:
            raise ValueError('Start time must be between 0 and the file\'s length')
        start_idx = bisect.bisect_left(self.t, start_time)
        
        if not self.lazy:
            yield from self.sxx[start_idx:, :]
        else:
            for idx in range(start_idx, self.sxx.shape[1]):
                binned = numpy.zeros(len(note_ranges))
                for i, (lo, hi) in enumerate(note_ranges):
                    lo_idx = bisect.bisect_left(self.f, lo)
                    hi_idx = bisect.bisect_left(self.f, hi)
                    binned[i] = numpy.sum(self.sxx[lo_idx:hi_idx, idx])

                oldmax = self.maxamp
                self.maxamp = numpy.max(numpy.abs(binned) * 1.05, initial=self.maxamp)
                if self.maxamp != oldmax:
                    print(f'New max: {oldmax} -> {self.maxamp}')
                yield binned
class App:
    samples_per_sec: int
    dframes_per_sample: int
    surface: pygame.surface.Surface
    paused: bool
    will_play: bool
    skip_frame_wait: bool
    sgram: Spectogram_Wrapper
    frame_iter: Iterator[numpy.typing.NDArray]
    bars: List[pygame.Rect]
    graph_area: pygame.Rect
    keyboard_area: pygame.Rect
    keys: List[Tuple[Iterable[Tuple[pygame.Rect, int]], int]]
    prog_bar: pygame.Rect
    prog_time: float
    last_amps: numpy.typing.NDArray

    def __init__(self, samples_per_sec: int, dframes_per_sample: int) -> None:
        self.paused = True
        self.samples_per_sec = samples_per_sec
        self.will_play = False
        self.skip_frame_wait = False
        self.dframes_per_sample = dframes_per_sample
        self.prog_time = 0
        self.sgram = None

    def construct(self, surface: pygame.surface.Surface) -> None:
        self.surface = surface
        self.surface.fill(WHITE)
        self.init_bars()
    
    def get_data(self, filename: str) -> None:
        self.sgram = Spectogram_Wrapper(filename, self.samples_per_sec)
    
    def __amp_interpolator(self, sample_iter: Iterator[numpy.typing.NDArray], interpolated_frames: int) -> Iterator[numpy.typing.NDArray]:
        if interpolated_frames == 0:
            yield from sample_iter
            return
        prev_frame = next(sample_iter)
        try:
            next_frame = next(sample_iter)
        except StopIteration:
            next_frame = prev_frame
        
        yield prev_frame
        idx = 1
        diffs = (next_frame - prev_frame) / (interpolated_frames + 1)
        while True:
            idx = (idx + 1) % (interpolated_frames + 1)
            if idx == 1:
                prev_frame = next_frame

                try:
                    next_frame = next(sample_iter)
                    diffs = (next_frame - prev_frame) / (interpolated_frames + 1)
                    yield prev_frame
                except StopIteration:
                    yield prev_frame
                    break
            else:
                yield prev_frame + idx * diffs
    
    def init_bars(self) -> None:
        win_width, win_height = self.surface.get_size()
        white_width, white_height_total = win_width // (7 * 7 + 3), int(win_height * 0.15)
        black_width, black_height = int((18/31) * white_width), int(0.5 * white_height_total)
        l_padding = (win_width - (white_width * (7 * 7 + 3))) // 2
        prog_bar_area_height = int(win_height * 0.05)
        bar_height = win_height - white_height_total - prog_bar_area_height

        self.prog_bar = pygame.Rect(l_padding, bar_height + white_height_total + int(0.25 * prog_bar_area_height), white_width * (7 * 7 + 3), prog_bar_area_height // 2)

        # 0 = C/F, 1 = black, 2 = D/G/A, 3 = E/B, 4 = C8
        key_types = [0, 1, *(a for _ in range(7) for a in (3, 0, 1, 2, 1, 3, 0, 1, 2, 1, 2, 1)), 3, 4]
        x = l_padding
        sensitivity_floor = self.samples_per_sec / (2**(1/12) - 1)

        self.keys = []
        for i, t in enumerate(key_types):
            if t == 0:
                self.keys.append(
                    ((
                        (pygame.Rect(x, bar_height, white_width - (black_width // 2), black_height), 1),
                        (pygame.Rect(x, bar_height + black_height, white_width, white_height_total - black_height), 2)
                    ), WHITE if sensitivity_floor <= note_ranges[i][1] else GRAY)
                )
            elif t == 1:
                self.keys.append((((pygame.Rect(x - (black_width // 2), bar_height, white_width - 2 * (black_width // 2), black_height), 0),), BLACK))
            elif t == 2:
                self.keys.append(
                    ((
                        (pygame.Rect(x + (black_width // 2) - (black_width % 2), bar_height, white_width - black_width + 2 * (black_width % 2), black_height), 1),
                        (pygame.Rect(x, bar_height + black_height, white_width, white_height_total - black_height), 2)
                    ), WHITE if sensitivity_floor <= note_ranges[i][1] else GRAY)
                )
            elif t == 3:
                self.keys.append(
                    ((
                        (pygame.Rect(x + (black_width // 2) - (black_width % 2), bar_height, white_width - (black_width // 2) + (black_width % 2), black_height), 1),
                        (pygame.Rect(x, bar_height + black_height, white_width, white_height_total - black_height), 2)
                    ), WHITE if sensitivity_floor <= note_ranges[i][1] else GRAY)
                )
            elif t == 4:
                self.keys.append((((pygame.Rect(x, bar_height, white_width, white_height_total), 0),), WHITE))
            
            if t != 1:
                x += white_width
        
        self.bars = [pygame.Rect(k[0][0].x, 0, k[0][0].width, bar_height) for k, _ in self.keys]

        self.graph_area = pygame.Rect(self.bars[0].left, 0, self.bars[-1].right - self.bars[0].left, bar_height)
        self.keyboard_area = pygame.Rect(self.keys[0][0][-1][0].left, bar_height, self.keys[-1][0][-1][0].right - self.keys[0][0][-1][0].left, white_height_total)
    
    def update_key_color(self, key_info: Tuple[Iterable[pygame.Rect], int], color: Tuple[int, int, int]) -> None:
            for k, t in key_info:
                if t == 0:
                    pygame.draw.rect(self.surface, color, (k.x + 1, k.y + 1, k.width - 2, k.height - 2))
                elif t == 1:
                    pygame.draw.rect(self.surface, color, (k.x + 1, k.y + 1, k.width - 2, k.height - 1))
                elif t == 2:
                    pygame.draw.rect(self.surface, color, (k.x + 1, k.y, k.width - 2, k.height - 1))

    def reset_keyboard(self) -> None:
        for r, c in self.keys:
            for k, t in r:
                pygame.draw.rect(self.surface, BLACK, k, width=1)
            self.update_key_color(r, c)

    def reset_bars(self) -> None:
        for b in self.bars:
            b.height = 0

    def select_file(self) -> None:
        self.skip_frame_wait = True
        filename = filedialog.askopenfilename()
        print('selected file', filename)

        try:
            pygame.mixer.music.load(filename)
        except:
            self.skip_frame_wait = False
            return
        self.get_data(filename)
        self.frame_iter = self.__amp_interpolator(self.sgram.iterator(), self.dframes_per_sample - 1)
        self.init_bars()
        self.surface.fill(WHITE)
        self.reset_keyboard()
        self.reset_bars()

        self.will_play = True

    def process_key(self, event) -> bool:
        if event.key == K_o and event.mod & (KMOD_LCTRL | KMOD_RCTRL) != 0:
            if not self.paused:
                pygame.mixer.music.pause()
                self.paused = True
            self.select_file()
        elif event.key == K_k or event.key == K_SPACE:
            if self.paused:
                pygame.mixer.music.unpause()
                self.paused = False
            else:
                pygame.mixer.music.pause()
                self.paused = True
        elif event.key == K_q and event.mod & (KMOD_LCTRL | KMOD_RCTRL) != 0:
            return True
        return False

    def process_mouse_click(self, event: pygame.event.EventType) -> bool:
        if self.prog_bar.collidepoint(event.pos):
            new_prog = (event.pos[0] - self.prog_bar.x) / self.prog_bar.width
            new_time = self.sgram.t[bisect.bisect_left(self.sgram.t, (self.sgram.length * new_prog))]

            try:
                pygame.mixer.music.set_pos(0) # Test if file supports seek
                self.prog_time = new_time
                self.frame_iter = self.__amp_interpolator(self.sgram.iterator(new_time), self.dframes_per_sample - 1)
                pygame.mixer.music.play(start=new_time)
                if self.paused:
                    pygame.mixer.music.pause()
            except Exception as e:
                print(f'Seeking not supported by current file type: {type(e)}')
                return False
        return False

    def process_events(self, events: List[pygame.event.Event]) -> bool:
        for e in events:
            if e.type == QUIT:
                return True
            elif e.type == KEYDOWN:
                if self.process_key(e):
                    return True
            elif e.type == MOUSEBUTTONDOWN:
                if self.process_mouse_click(e):
                    return True
        return False
    
    def update(self) -> bool:
        if self.will_play:
            pygame.mixer.music.play()
            self.paused = False
            self.is_blocked = False
            self.last_amps = numpy.zeros(len(note_ranges))
            self.prog_time = 0
            self.will_play = False

        if not self.paused:
            try:
                amps = next(self.frame_iter)
                #diffs = amps - self.last_amps
                self.last_amps = amps
            except StopIteration:
                self.surface.fill(WHITE, self.graph_area)
                self.reset_keyboard()
                self.reset_bars()
                self.paused = True
                return True
            
            total_amp = max(numpy.sum(amps), 1e-3)
            #for a, d, b, (keys, kcolor) in zip(amps, diffs, self.bars, self.keys):
            #self.surface.fill(WHITE, self.graph_area)
            for a, b, (keys, kcolor) in zip(amps, self.bars, self.keys):
                amp_prop = a / self.sgram.maxamp
                assert amp_prop >= 0 and amp_prop <= 1
                new_height = int(self.graph_area.height * amp_prop)

                if new_height > b.height:
                    self.surface.fill(BLUE, (b.x, self.graph_area.height - new_height, b.width, new_height - b.height))
                elif new_height < b.height:
                    self.surface.fill(WHITE, (b.x, self.graph_area.height - b.height, b.width, b.height - new_height))
                b.height = new_height
                #self.surface.fill(BLUE, (b.x, self.graph_area.height - new_height, b.width, new_height))

                sound_prop = a / total_amp
                #delta_prop = a / (a - d) if a - d != 0 else 100
                significance = sound_prop
                weight = sound_prop
                self.update_key_color(keys, color_interpolate(weight if significance >= 0.06 else 0, 0, 0.5, [kcolor, GREEN, YELLOW, RED]))

            self.prog_time += 1 / (self.samples_per_sec * self.dframes_per_sample)
        
        if self.sgram is not None:
            self.surface.fill(LIGHT_GRAY, self.prog_bar)
            prog = self.prog_time / self.sgram.length
            self.surface.fill(LIGHT_BLUE, (self.prog_bar.x, self.prog_bar.y, self.prog_bar.width * prog, self.prog_bar.height))

        return True

def quit() -> None:
    pygame.quit()
    sys.exit()

def main() -> None:
    samples_per_sec = 8
    target_dfps = 90

    dframes_per_sec = target_dfps - (target_dfps % samples_per_sec)
    assert dframes_per_sec % samples_per_sec == 0
    app = App(samples_per_sec, dframes_per_sec // samples_per_sec)

    root = tk.Tk()
    root.withdraw()

    pygame.init()
    DSURF = pygame.display.set_mode((1000, 500))
    pygame.display.set_caption("MusicVis")
    app.construct(DSURF)

    ns_per_frame = int(1e9 / dframes_per_sec)
    next_frame_time = time.perf_counter_ns() + ns_per_frame

    while True:
        if not app.update():
            quit()
        pygame.display.flip()

        if app.process_events(pygame.event.get()):
            quit()
        
        if app.skip_frame_wait:
            app.skip_frame_wait = False
            next_frame_time = time.perf_counter_ns() + ns_per_frame
            continue

        while time.perf_counter_ns() < next_frame_time:
            pass
        next_frame_time += ns_per_frame

if __name__ == "__main__":
    main()