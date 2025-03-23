import cv2
import numpy as np

class VideoFileInput:
    def __init__(self, filename):
        self.filename = filename
        self.cap = cv2.VideoCapture(filename)

    def read(self):
        ret, frame = self.cap.read()
        return frame

    def release(self):
        self.cap.release()

def process(frame, audio_buffer):
    #calculate energy content of audio_buffer
    audiofft = np.fft.fft(audio_buffer)
    energy = np.sum(np.abs(audiofft)) + 0.1

    bass = np.sum(np.abs(audiofft[0:100])) + 0.1
    print('Bass: ' + str(bass))
    middle = np.sum(np.abs(audiofft[100:1000])) + 0.1
    print('Middle: ' + str(middle))
    treble = np.sum(np.abs(audiofft[1000:])) + 0.1
    print('Treble: ' + str(treble))

    print('Energy: ' + str(energy))
    #apply brightness to frame
    for i in range(0, frame.shape[0]):
        for j in range(0, frame.shape[1]):
            frame[i][j][0] = frame[i][j][0] * energy + bass
            frame[i][j][1] = frame[i][j][1] * energy + middle
            frame[i][j][2] = frame[i][j][2] * energy + treble

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return None

    return frame

class VideoFileOutput:
    def __init__(self, filename, fps, size):
        self.filename = filename
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(filename, self.fourcc, fps, size)

    def write(self, frame):
        self.out.write(frame)

    def release(self):
        self.out.release()

import moviepy as mp

def extract_audio(video_filename):
    video = mp.VideoFileClip(video_filename)
    return video.audio

def main():
    filename = '../japiCum/japiCumRend02.mov'
    vfi = VideoFileInput(filename)
    audio = extract_audio(filename)
    fps = vfi.cap.get(cv2.CAP_PROP_FPS)
    size = (int(vfi.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vfi.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vfo = VideoFileOutput('japiCumVFX01.avi', fps, size)
    while True:
        print('Reading frame ' + str(vfi.cap.get(cv2.CAP_PROP_POS_FRAMES)))
        frame = vfi.read()
        audio_buffer_length = 44100//fps
        audio_buffer_start = vfi.cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        audio_buffer = audio.subclipped(audio_buffer_start, audio_buffer_start + audio_buffer_length).to_soundarray(fps)
        frame = process(frame, audio_buffer)
        if frame is None:
            print('End of video')
            break
        vfo.write(frame)
    vfi.release()
    vfo.release()

if __name__ == '__main__':
    main()
