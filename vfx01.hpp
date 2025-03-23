#include <opencv2/opencv.hpp>
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

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/opt.h>
    #include <libavutil/samplefmt.h>
    #include <libavutil/channel_layout.h>
    #include <libavutil/audio_fifo.h>
    #include <libavutil/avassert.h>
    #include <libavutil/avstring.h>
    #include <libavutil/opt.h>
    #include <libavutil/mathematics.h>
}

using namespace std;
using namespace cv;

class VideoInputFile {
public:
    VideoInputFile(const string& filename);
    ~VideoInputFile();
    bool read(Mat& frame);
    int getFrameWidth() const;
    int getFrameHeight() const;
    double getFrameRate() const;
    int getFrameCount() const;
    int getFourcc() const;
    int getCodec() const;
    string getCodecName() const;
    string getFilename() const;
    double getDuration() const;
    bool isOpened() const;
    void release();
private:
    VideoCapture cap;
    string filename;
    int frame_width;
    int frame_height;
    double frame_rate;
    int frame_count;
    int fourcc;
    int codec;
    string codec_name;
};

class VideoOutputFile {
public:
    VideoOutputFile(const string& filename, int fourcc, double frame_rate, int frame_width, int frame_height);
    ~VideoOutputFile();
    void write(const Mat& frame);
    void release();
private:
    VideoWriter writer;
};

void extractAudio(const string& filename, const string& audio_filename);

class VideoEffectBassMiddleTreble{
public:
    VideoEffectBassMiddleTreble();
    void apply(const Mat& input, Mat& output, double bass, double middle, double treble);
private:
    double bass;
    double middle;
    double treble;
};

class VideoEffectBrightness{
public:
    VideoEffectBrightness();
    void apply(const Mat& input, Mat& output, double brightness);
private:
    double brightness;
};

void show_frame(const Mat& frame);
