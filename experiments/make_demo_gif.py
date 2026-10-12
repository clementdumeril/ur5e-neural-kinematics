"""
Turns the Webots movie into the README GIF.

    python experiments/make_demo_gif.py [results/webots/capture.mp4]

Skips the first START_S seconds, speeds the rest up so that it lasts about
TARGET_S seconds, then encodes a
720-px-wide, 12 fps GIF with an optimised palette into assets/demo.gif. Uses
the ffmpeg shipped with Webots unless FFMPEG points to another binary.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FFMPEG = os.environ.get('FFMPEG', os.path.join(
    os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Webots', 'msys64', 'mingw64', 'bin', 'ffmpeg.exe'))
TARGET_S = float(os.environ.get('GIF_SECONDS', 9.0))
WIDTH, FPS = 720, 12
START_S = float(os.environ.get('GIF_START', 4.5))   # the arm starts upright, out of frame


def duration(path):
    out = subprocess.run([FFMPEG, '-i', path], capture_output=True, text=True).stderr
    h, m, s = re.search(r'Duration: (\d+):(\d+):([\d.]+)', out).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results', 'webots', 'capture.mp4')
    dst = os.path.join(ROOT, 'assets', 'demo.gif')
    d = duration(src) - START_S
    speed = max(1.0, d / TARGET_S)
    vf = (f"setpts=PTS/{speed:.4f},fps={FPS},scale={WIDTH}:-1:flags=lanczos,"
          "split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle")
    subprocess.run([FFMPEG, '-y', '-loglevel', 'error', '-ss', f'{START_S}', '-i', src,
                    '-vf', vf, '-loop', '0', dst],
                   check=True)
    size = os.path.getsize(dst) / 1e6
    print(f"{src}: {d:.1f} s -> x{speed:.2f} -> {dst} ({d / speed:.1f} s, {size:.1f} MB)")

    # Four stills for the paper: reading pose, grasp, transport, cube on the tray.
    strip = os.path.join(ROOT, 'paper', 'figures', 'webots_sequence.png')
    times = [START_S + f * d for f in (0.0, 0.42, 0.68, 0.98)]
    sel = '+'.join(f"eq(n,{int(t * 25)})" for t in times)
    subprocess.run([FFMPEG, '-y', '-loglevel', 'error', '-i', src, '-vf',
                    f"select='{sel}',scale=640:-1,tile=4x1:padding=6:color=white",
                    '-frames:v', '1', strip], check=True)
    print(f"stills: {strip}")


if __name__ == '__main__':
    main()
