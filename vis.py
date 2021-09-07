from scipy.io import wavfile
import numpy
import matplotlib.pyplot as plot
import matplotlib.patches as patches
import matplotlib.figure as figure
import matplotlib.axes as axes
from functools import reduce
import subprocess
import sys
import time
import multiprocessing

class FFT_Wrapper:        
    def __init__(self, file_name):
        self.file_path = f"wavs/{file_name}.wav"
        self.samplerate, self.raw_audio = wavfile.read(self.file_path)
    
    def compute_fft(self, start_frame, end_frame):
        n = end_frame - start_frame + 1
        
        fft_l = numpy.abs(numpy.fft.rfft(self.raw_audio[start_frame:end_frame + 1, 0], n=n)[1:])
        fft_r = numpy.abs(numpy.fft.rfft(self.raw_audio[start_frame:end_frame + 1, 1], n=n)[1:])
        
        return (fft_l, fft_r)
        
    def iterator(self, step_size, s_frame=0, e_frame=None):
        if e_frame is None:
            e_frame = self.raw_audio.shape[0]
        
        while s_frame < e_frame:
            l = numpy.abs(numpy.fft.rfft(self.raw_audio[s_frame:s_frame + step_size, 0], n=step_size)[1:])
            r = numpy.abs(numpy.fft.rfft(self.raw_audio[s_frame:s_frame + step_size, 1], n=step_size)[1:])
            yield (l, r)
            s_frame += step_size
    
    def generate_stereo(self, sample_fps, display_fps):
        frames_per_frame = self.samplerate // sample_fps
        interpolated_frames_per_frame = display_fps // sample_fps
        
        prev_frame = numpy.zeros((2, frames_per_frame // 2))
        
        for i, (l, r) in enumerate(self.iterator(frames_per_frame)):
            cur_frame = numpy.array((l, r))
            
            yield from self.__interpolate(prev_frame, cur_frame, interpolated_frames_per_frame)
            prev_frame = cur_frame
    
    def generate_mono(self, sample_fps, display_fps, s_frame=0, e_frame=None):
        frames_per_frame = self.samplerate // sample_fps
        interpolated_frames_per_frame = display_fps // sample_fps
        
        prev_frame = numpy.zeros(frames_per_frame // 2)
        
        for i, (l, r) in enumerate(self.iterator(frames_per_frame, s_frame, e_frame)):
            cur_frame = numpy.zeros(prev_frame.shape)
            
            for i in range(l.shape[0]):
                cur_frame[i] = (l[i] + r[i]) / 2
            
            yield from self.__interpolate(prev_frame, cur_frame, interpolated_frames_per_frame)
            prev_frame = cur_frame
    
    def __interpolate(self, first, second, n):
        diff = (second - first) / n
        
        for i in range(0, n):
            yield first + i * diff

class Histogram_Worker:
    def __init__(self, fft_iterator, sample_fps, display_fps, bins, fft_bin_to_tone_bin_map, max_amp):
        self.figure = figure.Figure()
        self.axis = self.figure.add_subplot()
        
        self.fft_iterator = fft_iterator
        self.sample_fps = sample_fps
        self.display_fps = display_fps
        self.bins = bins
        self.max_amp = max_amp
        self.global_amp_threshold = max_amp * 0.05
        self.fft_bin_to_tone_bin_map = fft_bin_to_tone_bin_map
        
        self.bars = self.axis.hist([0], bins=bins)[2]
        
        black, white = (0, 0, 0), (1, 1, 1)
        key_types = [0, *([1, 2, 0, 1, 3, 1, 2, 0, 1, 3, 1, 3] * 8)[0:86], 4]  # 0=C, 1=C#, 2=B, 3=D, 4=C8
        key_height = (max_amp * 0.1, max_amp * 0.05)
        key_generators = [
            lambda i: patches.Rectangle((self.bars[i].get_x(), -key_height[0]), self.bars[i].get_width() + self.bars[i + 1].get_width() / 2, key_height[0], facecolor=white, edgecolor=black),
            lambda i: patches.Rectangle((self.bars[i].get_x(), -key_height[1]), self.bars[i].get_width(), key_height[1], facecolor=black, edgecolor=black),
            lambda i: patches.Rectangle((self.bars[i].get_x() - self.bars[i - 1].get_width() / 2, -key_height[0]), self.bars[i].get_width() + self.bars[i - 1].get_width() / 2, key_height[0], facecolor=white, edgecolor=black),
            lambda i: patches.Rectangle((self.bars[i].get_x() - self.bars[i - 1].get_width() / 2, -key_height[0]), self.bars[i].get_width() + self.bars[i - 1].get_width() / 2 + self.bars[i + 1].get_width() / 2, key_height[0], facecolor=white, edgecolor=black),
            lambda i: patches.Rectangle((self.bars[i].get_x(), -key_height[0]), self.bars[i].get_width(), key_height[0], facecolor=white, edgecolor=black)
        ]
        self.key_rects = [(key_generators[key_types[i]](i), key_types[i]) for i in range(0, 88)]
        
        self.axis.set_xscale("log", base=2)
        self.axis.set_xlim((bins[0], bins[-1]))
        self.axis.set_ylim((-max_amp * 0.1, max_amp * 1.05))
        self.axis.get_xaxis().set_visible(False)
        self.axis.get_yaxis().set_visible(False)
        
        for rect in (*(r for r, t in self.key_rects if not t == 1), *(r for r, t in self.key_rects if t == 1)):
            self.axis.add_patch(rect)
        
        self.bbox_extent = self.axis.get_window_extent().transformed(self.figure.dpi_scale_trans.inverted())
    
    def bin_sums(self, arr):
        binned_arr = numpy.zeros(len(self.bins) - 1)
        
        for i, amp in enumerate(arr):
            if self.fft_bin_to_tone_bin_map[i] >= 0:
                binned_arr[self.fft_bin_to_tone_bin_map[i]] += amp
        
        return binned_arr
    
    def color_interpolate(val, min, max, colors=[(0, 1, 0), (1, 1, 0), (1, 0, 0)]):
        if val == max:
            return colors[-1]
        
        proportion = (val - min) / (max - min)
        i = int((len(colors) - 1) * proportion)
        low, high = colors[i], colors[i + 1]
        n_prop = (proportion - (i / (len(colors) - 1))) * (len(colors) - 1)
        return (low[0] + (high[0] - low[0]) * n_prop, low[1] + (high[1] - low[1]) * n_prop, low[2] + (high[2] - low[2]) * n_prop)
    
    def start(self, frame_offset):
        for i, data in enumerate(self.fft_iterator):
            binned = self.bin_sums(data)
            amp_threshold = numpy.amax(numpy.amax(binned) * 0.5, initial=self.global_amp_threshold)
            
            for bin_amp, bar, (rect, type) in zip(binned, self.bars, self.key_rects):
                bar.set_height(bin_amp)
                rect.set_facecolor(Histogram_Worker.color_interpolate(bin_amp, 0, self.max_amp) if bin_amp >= amp_threshold else ((0, 0, 0) if type == 1 else (1, 1, 1)))
            
            self.figure.savefig(f"temp/img_out/{int(frame_offset + i)}.png", bbox_inches=self.bbox_extent, pad_inches=0)

class Image_Vis:
    def __init__(self, file_name):
        self.fft = FFT_Wrapper(file_name)
        
    def line_plot(self, sample_fps, display_fps):
        figure, axis = plot.subplots()
        frames_per_frame = self.fft.samplerate // sample_fps
        interpolated_frames_per_frame = display_fps // sample_fps
        out_frames = int(numpy.ceil(self.fft.raw_audio.shape[0] / frames_per_frame)) * interpolated_frames_per_frame
        
        self.__clean_out_dir()
        
        print(f"Generating from {self.fft.file_path} at {sample_fps} audio sample(s) per second, {display_fps} display frames per second")
        
        self.__set_up_axes(axis, self.fft.samplerate // 2)
        x_vals = numpy.linspace(0, self.fft.samplerate // 2, frames_per_frame // 2)
        left_line = axis.plot(x_vals, numpy.zeros(len(x_vals)), "r")[0]
        right_line = axis.plot(x_vals, numpy.zeros(len(x_vals)), "b")[0]
        
        max_amp = reduce(lambda prev, next: numpy.amax(next, initial=prev), self.fft.iterator(frames_per_frame), 0)
        print("First pass completed")
        axis.set_ylim((0, max_amp))
        
        for i, data in enumerate(self.fft.generate_stereo(sample_fps, display_fps)):
            left_line.set_ydata(data[0])
            right_line.set_ydata(data[1])
            
            figure.savefig(f"temp/img_out/{i}.png")
            print(f"Completed frame {i + 1} of {out_frames}", end="\r")
        
        print("\nMerging...")
        self.__ffmpeg(display_fps)
        print("Output written to \"temp/out.mp4\"")
    
    def histogram(self, sample_fps, display_fps, num_workers=1):
        self.__clean_out_dir()
        
        step = numpy.power(2, 1 / 12)
        bins = [27.5 * (step ** -0.5), *(t * (step ** 0.5) for t in (27.5 * (step ** s) for s in range(0, 88)))]
        
        def binary_search(arr, x):
            low = 0
            high = len(arr) - 1
         
            while low <= high: 
                mid = (high + low) // 2

                if arr[mid] < x:
                    low = mid + 1
                elif arr[mid] > x:
                    high = mid - 1
                else:
                    return mid if mid < len(arr) - 1 else -1
         
            return high if high < len(arr) - 1 else -1
        
        fft_bin_to_tone_bin_map = [binary_search(bins, sample_fps / 2 + sample_fps * f) for f in range(self.fft.samplerate // sample_fps - 1)]
        
        def bin_sums(arr):
            binned_arr = numpy.zeros(len(bins) - 1)
            
            for i, amp in enumerate(arr):
                if fft_bin_to_tone_bin_map[i] >= 0:
                    binned_arr[fft_bin_to_tone_bin_map[i]] += amp
            
            return binned_arr
        
        print("Begin first pass")
        max_amp = reduce(lambda prev, next: numpy.amax(next, initial=prev), (bin_sums((l + r) / 2) for l, r in self.fft.iterator(self.fft.samplerate // sample_fps)), 0)
        print("First pass completed")
        
        out_audio_frames = int(numpy.ceil(self.fft.raw_audio.shape[0] * sample_fps / self.fft.samplerate))
        out_display_frames = int(numpy.ceil(self.fft.raw_audio.shape[0] / (self.fft.samplerate / display_fps)))
        
        worker_frame_load = [*(numpy.ceil(out_audio_frames / num_workers) for i in range(0, num_workers - 1)), out_audio_frames - numpy.ceil(out_audio_frames / num_workers) * (num_workers - 1)]
        worker_frame_offsets = [0, *numpy.cumsum(worker_frame_load)][:-1]
        
        frames_per_frame = self.fft.samplerate // sample_fps
        workers = [Histogram_Worker(
            [*self.fft.generate_mono(sample_fps, display_fps, int(frame_offset * frames_per_frame), int((frame_offset + frame_count) * frames_per_frame))],
            sample_fps,
            display_fps,
            [*bins],
            [*fft_bin_to_tone_bin_map],
            max_amp
        ) for i, (frame_count, frame_offset) in enumerate(zip(worker_frame_load, worker_frame_offsets))]
        worker_processes = [multiprocessing.Process(target=worker.start, args=(offset * (display_fps // sample_fps),)) for worker, offset in zip(workers, worker_frame_offsets)]
        
        print(f"Starting, using {num_workers} workers")
        
        s_time = time.time()
        for worker_process in worker_processes:
            worker_process.start()
        for worker_process in worker_processes:
            worker_process.join()
        print(f"Completed in {time.time() - s_time} seconds")
        
        print("Merging...")
        self.__ffmpeg(display_fps)
        print("Output written to \"temp/out.mp4\"")
    
    def __ffmpeg(self, display_fps):
        subprocess.run(["ffmpeg", "-framerate", str(display_fps), "-loglevel", "warning", "-thread_queue_size", "256", "-i", "temp/img_out/%d.png", "-ac", "2", "-i", self.fft.file_path, "-y", "temp/out.mp4"])
    
    def __set_up_axes(self, axis, max_freq):
        axis.set_xscale("log", base=2)
        ticks = list(self.__octave_freq_generator(max_freq))
        axis.set_xticks([t[0] for t in ticks])
        axis.set_xticklabels([t[1] for t in ticks])
        axis.set_xlim(left=27.5)
        axis.get_yaxis().set_visible(False)
        
    def __octave_freq_generator(self, max_freq):
        f = 27.5 * numpy.power(2, 3 / 12)   #C1
        i = 1
        while f <= max_freq:
            yield (f, f"C{i}")
            f *= 2
            i += 1
    
    def __clean_out_dir(self):
        import os
        for filename in os.listdir("temp/img_out"):
            path = os.path.join("temp/img_out", filename)
            if (os.path.isfile(path)):
                os.unlink(path)

if __name__ == "__main__":
    vis = Image_Vis(sys.argv[1])
    #vis.line_plot(int(sys.argv[2]), int(sys.argv[3]))
    vis.histogram(int(sys.argv[2]), int(sys.argv[3]), 4)
