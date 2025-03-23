#include "vfx01.hpp"
#include "/usr/local/include/fftw3.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <SFML/Audio.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>

using namespace std;
using namespace cv;


VideoInputFile::VideoInputFile(const string& filename) {
    cap.open(filename);
    if (!cap.isOpened()) {
        cerr << "Error: cannot open video file " << filename << endl;
        return;
    }
    this->filename = filename;
    frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    frame_rate = cap.get(CAP_PROP_FPS);
    frame_count = cap.get(CAP_PROP_FRAME_COUNT);
    fourcc = cap.get(CAP_PROP_FOURCC);
    codec = cap.get(CAP_PROP_FOURCC);
    codec_name = cap.get(CAP_PROP_FOURCC);
}

VideoInputFile::~VideoInputFile() {
    this->cap.release();
}

bool VideoInputFile::read(Mat& frame) {
    return cap.read(frame);
}

int VideoInputFile::getFrameWidth() const {
    return frame_width;
}

int VideoInputFile::getFrameHeight() const {
    return frame_height;
}

int VideoInputFile::getFourcc() const {
    return this->fourcc;
}

int VideoInputFile::getCodec() const {
    return this->codec;
}

string VideoInputFile::getCodecName() const {
    return this->codec_name;
}

string VideoInputFile::getFilename() const {
    return this->filename;
}

int VideoInputFile::getFrameCount() const {
    return this->frame_count;
}

double VideoInputFile::getFrameRate() const {
    return this->frame_rate;
}

double VideoInputFile::getDuration() const {
    return frame_count / frame_rate;
}

VideoOutputFile::VideoOutputFile(const string& filename, int fourcc, double frame_rate, int frame_width, int frame_height) {
    writer.open(filename, fourcc, frame_rate, Size(frame_width, frame_height));
    if (!writer.isOpened()) {
        cerr << "Error: cannot open video file " << filename << endl;
        return;
    }
}

VideoOutputFile::~VideoOutputFile() {
    release();
}

void VideoOutputFile::write(const Mat& frame){
    writer.write(frame);
}

void VideoOutputFile::release() {
    writer.release();
}

VideoEffectBassMiddleTreble::VideoEffectBassMiddleTreble() {
    bass = 0.0;
    middle = 0.0;
    treble = 0.0;
}

void VideoEffectBassMiddleTreble::apply(const Mat& input, Mat& output, double bass, double middle, double treble) {
    Mat channels[3];
    split(input, channels);
    cout << "bass: " << bass << endl;
    cout << "middle: " << middle << endl;
    cout << "treble: " << treble << endl;
    channels[0] *= bass;
    channels[1] *= middle;
    channels[2] *= treble;
    merge(channels, 3, output);
}

VideoEffectBrightness::VideoEffectBrightness() {
    brightness = 0.0;
}

void VideoEffectBrightness::apply(const Mat& input, Mat& output, double brightness) {
    Mat channels[3];
    split(input, channels);
    cout << "brightness: " << brightness << endl;
    channels[0] *= brightness;
    channels[1] *= brightness;
    channels[2] *= brightness;
    merge(channels, 3, output);
}

void extract_audio(const string& filename, std::vector<double>& audiobuffer, int& sample_rate) {
    sf::SoundBuffer buffer;
    if (!buffer.loadFromFile(filename)) {
        cerr << "Error: cannot load audio file " << filename << endl;
        return;
    }

    const sf::Int16* samples = buffer.getSamples();
    size_t sample_count = buffer.getSampleCount();
    sample_rate = buffer.getSampleRate();

    audiobuffer.resize(sample_count);
    for (size_t i = 0; i < sample_count; ++i) {
        audiobuffer[i] = samples[i] / 32768.0; // Normalize to range [-1, 1]
    }

    cout << "Sample rate: " << sample_rate << endl;
    cout << "Total samples in audiobuffer: " << audiobuffer.size() << endl;
}

void show_frame(const Mat& frame) {
    imshow("Frame", frame);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <video file> <audio file>" << endl;
        return 1;
    }

    namedWindow("Frame", WINDOW_NORMAL);
    VideoInputFile input(argv[1]);
    VideoOutputFile output("japiCumVFX01.avi", input.getFourcc(), input.getFrameRate(), input.getFrameWidth(), input.getFrameHeight());
    int sample_rate;
    std::vector<double> audiobuffer;

    cout << "Extracting audio from " << argv[2] << endl;
    extract_audio(argv[2], audiobuffer, sample_rate);
    cout << "Audio extracted" << endl;

    cout << "Sample rate: " << sample_rate << endl;
    cout << "Total samples in audiobuffer: " << audiobuffer.size() << endl;
    int frame_amount = input.getFrameCount();
    int frame_index = 0;
    int sample_index = 0;
    int samples_per_frame = sample_rate / input.getFrameRate();
    cout << "Samples per frame: " << samples_per_frame << endl;
    std::vector<double> samples(samples_per_frame);

    VideoEffectBassMiddleTreble vfx1;
    VideoEffectBrightness vfx2;

    Mat frame;
    fftw_complex fftw_output[samples_per_frame];

    cout << "Processing video" << endl;
    while (input.read(frame)) {
        frame_index++;
        sample_index += samples_per_frame;

        cout << "Processing frame " << frame_index << " of " << frame_amount << endl;
        Mat output_frame;

        cout << "copying audio samples" << endl;
        int start_index = sample_index;
        if (start_index + samples_per_frame >= audiobuffer.size()) {
            sample_index = 0;
            start_index = 0;
        }
        for (int i = 0; i < samples_per_frame; i++) {
            samples[i] = audiobuffer[start_index + i];
        }

        cout << "applying fftw" << endl;
        fftw_plan plan = fftw_plan_dft_r2c_1d(samples_per_frame, samples.data(), fftw_output, FFTW_ESTIMATE);
        fftw_execute(plan);

        double bass = 0.0;
        double middle = 0.0;
        double treble = 0.0;
        double brightness = 0.0;
        
        for (int i = 0; i < samples_per_frame / 2; i++) {
            double freq = i * sample_rate / samples_per_frame;
            double magnitude = sqrt(fftw_output[i][0] * fftw_output[i][0] + fftw_output[i][1] * fftw_output[i][1]);
            if (freq < 500) {
                bass += magnitude;
            } else if (freq < 1500) {
                middle += magnitude;
            } else {
                treble += magnitude;
            }
            brightness += magnitude;
        }
        bass /= samples_per_frame / 1.2;
        middle /= samples_per_frame / 1.5;
        treble /= samples_per_frame / 12.5;
        brightness /= samples_per_frame / 2.0;

        cout << "bass: " << bass << endl;
        cout << "middle: " << middle << endl;
        cout << "treble: " << treble << endl;
        cout << "brightness: " << brightness << endl;

        vfx1.apply(frame, output_frame, bass, middle, treble);
        vfx2.apply(frame, output_frame, brightness);
        output.write(output_frame);
        show_frame(output_frame);
        waitKey(1);

        fftw_destroy_plan(plan);
    }

    return 0;
}
