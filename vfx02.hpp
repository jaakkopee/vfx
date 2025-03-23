#ifndef VFX01_HPP
#define VFX01_HPP

#include <SFML/Audio.hpp>
#include <SFML/Graphics.hpp>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <iostream>
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

class VideoInputFile {
public:
    VideoInputFile(const std::string& filename);
    ~VideoInputFile();
    bool read(cv::Mat& frame);
    int getFrameWidth() const;
    int getFrameHeight() const;
    double getFrameRate() const;
    int getFrameCount() const;
    int getFourcc() const;
    int getCodec() const;
    std::string getCodecName() const;
    std::string getFilename() const;
    double getDuration() const;
    bool isOpened() const;
    void release();
private:
    cv::VideoCapture cap;
    std::string filename;
    int frame_width;
    int frame_height;
    double frame_rate;
    int frame_count;
    int fourcc;
    int codec;
    std::string codec_name;
};

class MediaWriter {
private:
    const AVCodec* video_codec = nullptr;
    const AVCodec* audio_codec = nullptr;
    AVFormatContext* format_ctx = nullptr;
    AVCodecContext* video_codec_ctx = nullptr;
    AVCodecContext* audio_codec_ctx = nullptr;
    int video_stream_index = -1;
    int audio_stream_index = -1;
public:
    MediaWriter();
    ~MediaWriter();
    bool open(const std::string& filename);
    bool writeVideoFrame(AVFrame* frame);
    bool writeAudioBuffer(uint8_t* buffer, int size, int sample_rate, int channels);
};

class VideoEffectBassMiddleTreble {
public:
    VideoEffectBassMiddleTreble();
    void apply(const cv::Mat& input, cv::Mat& output, double bass, double middle, double treble);
private:
    double bass;
    double middle;
    double treble;
};

class VideoEffectBrightness {
public:
    VideoEffectBrightness();
    void apply(const cv::Mat& input, cv::Mat& output, double brightness);
private:
    double brightness;
};

class VideoEffectEdgeDetection {
public:
    VideoEffectEdgeDetection();
    void apply(const cv::Mat& input, cv::Mat& output, double blend);
};

class VideoEffectMotionBlur {
public:
    VideoEffectMotionBlur();
    void apply(const cv::Mat& input, cv::Mat& output, double amount);
private:
    cv::Mat kernel;
};

class VideoEffectCellAutoMata {
public:
    VideoEffectCellAutoMata();
    void apply(const cv::Mat& input, cv::Mat& output, double neighborhood_size, double threshold);
private:
    std::vector<std::vector<int>> neighborhood;
};

void extract_audio(const std::string& filename, std::vector<double>& audiobuffer, int& sample_rate);

#endif // VFX01_HPP
