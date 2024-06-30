# %%
import librosa
import numpy as np
from manim import *
from manim.opengl import *


def get_beats(filename):
    y, sr = librosa.load(filename)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return beat_times


class BeatCircles(Scene):
    CONFIG = {"include_sound": True}

    def __init__(self, beat_times, audio_file, **kwargs):
        super().__init__(**kwargs)
        self.beat_times = beat_times
        self.audio_file = audio_file

    def construct(self):
        self.add_sound(sound_file=self.audio_file, time_offset=0)
        circles = []
        for i, beat in enumerate(self.beat_times):
            # Generate random position
            random_position = np.array([np.random.uniform(-3.5, 3.5), np.random.uniform(-4, 4), 0])
            circle = Circle(radius=1, color=WHITE).set_opacity(0.5)
            circle.move_to(random_position)
            self.add(circle)
            circles.append(circle)

            # Animate the circle to appear and decrease in size
            self.play(circle.animate.set_opacity(1), run_time=0.1)
            self.wait(beat - (self.beat_times[i - 1] if i > 0 else 0))
            self.play(circle.animate.scale(0.1).set_opacity(0), run_time=0.5)


if __name__ == "__main__":
    audio_file = "/Users/matbreotten/Documents/code/StreamDiffusion/media/assets/song01.wav"
    from manim import config

    config.media_width = "100%"
    config.quality = "high_quality"
    config.sound = True
    config.max_files_cached = 300
    beat_times = get_beats(audio_file)
    scene = BeatCircles(beat_times, audio_file)
    scene.render(preview=True)
