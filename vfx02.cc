#include "vfx02.hpp"
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

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
}

MediaWriter::MediaWriter() {
    video_codec = nullptr;
    audio_codec = nullptr;
    format_ctx = nullptr;
    video_codec_ctx = nullptr;
    audio_codec_ctx = nullptr;
    video_stream_index = -1;
    audio_stream_index = -1;
}

MediaWriter::~MediaWriter() {
    if (format_ctx) {
        av_write_trailer(format_ctx);
        avformat_free_context(format_ctx);
    }
    // Free the codec context
    if (video_codec_ctx) {
        avcodec_free_context(&video_codec_ctx);
    }
    if (audio_codec_ctx) {
        avcodec_free_context(&audio_codec_ctx);
    }
    //close the output file
    if (format_ctx && format_ctx->pb) {
        avio_closep(&format_ctx->pb);
    }
}

bool MediaWriter::open(const string& filename) {
    cout << "Opening output file " << filename << endl;
    
    // Audio codec
    audio_codec = avcodec_find_encoder(AV_CODEC_ID_AAC);
    if (!audio_codec) {
        cerr << "Error: cannot find audio codec" << endl;
        return false;
    }

    // Video codec
    video_codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!video_codec) {
        cerr << "Error: cannot find video codec" << endl;
        return false;
    }

    // Allocate format context
    format_ctx = avformat_alloc_context();
    if (!format_ctx){
        cerr << "Error: cannot allocate format context" << endl;
        return false;
    }

    // Allocate video codec context
    video_codec_ctx = avcodec_alloc_context3(nullptr);
    if (!video_codec_ctx){
        cerr << "Error: cannot allocate video codec context" << endl;
        return false;
    }

    // Allocate audio codec context
    audio_codec_ctx = avcodec_alloc_context3(audio_codec);
    if (!audio_codec_ctx){
        cerr << "Error: cannot allocate audio codec context" << endl;
        return false;
    }

    // Set format
    const char* fn = (const char*)filename.c_str();
    format_ctx->oformat = av_guess_format(nullptr, fn, nullptr);
    if (!format_ctx->oformat){
        cerr << "Error: cannot guess format" << endl;
        return false;        
    }

    // Add video stream
    video_stream_index = avformat_new_stream(format_ctx, nullptr)->index;
    if (video_stream_index < 0){
        cerr << "Error: cannot add video stream" << endl;
        return false;
    }

    // Add audio stream
    audio_stream_index = avformat_new_stream(format_ctx, nullptr)->index;
    if (audio_stream_index < 0){
        cerr << "Error: cannot add audio stream" << endl;
        return false;
    }

    // Open output file
    if (avio_open(&format_ctx->pb, fn, AVIO_FLAG_WRITE) < 0){
        cerr << "Error: cannot open output file" << endl;
        return false;
    }

    return true;
}

bool send_frame(AVCodecContext* video_codec_ctx, AVFrame* frame) {
    // Validate inputs
    if (!video_codec_ctx || !frame) {
        cerr << "Error: Invalid null pointer passed to avcodec_send_frame" << endl;
        return false;
    }
    
    // Ensure frame is properly initialized
    if (!frame->data[0]) {
        cerr << "Error: Frame buffers not allocated" << endl;
        return false;
    }
    
    // Send the frame
    int ret = avcodec_send_frame(video_codec_ctx, frame);
    if (ret < 0) {
        char errbuf[128];
        av_strerror(ret, errbuf, sizeof(errbuf));
        cerr << "Error: Failed to send frame: " << errbuf << endl;
        return false;
    }
    
    return true;
}

bool MediaWriter::writeVideoFrame(AVFrame* frame) {
    cout << "Writing video frame" << endl;
    if (!frame || video_stream_index < 0) return false;

    AVPacket* pkt = av_packet_alloc();
    if (!pkt) return false;

    int ret = avcodec_receive_packet(video_codec_ctx, pkt);
    if (ret < 0) return false;

    pkt->stream_index = video_stream_index;
    pkt->pts = av_rescale_q(video_codec_ctx->frame_num, video_codec_ctx->time_base, format_ctx->streams[video_stream_index]->time_base);
    pkt->dts = pkt->pts;
    pkt->duration = av_rescale_q(1, video_codec_ctx->time_base, format_ctx->streams[video_stream_index]->time_base);

    ret = av_interleaved_write_frame(format_ctx, pkt);
    av_packet_unref(pkt);

    cout << "Video frame written" << endl;

    return ret >= 0;
}

bool MediaWriter::writeAudioBuffer(uint8_t* buffer, int size, int sample_rate, int channels) {
    cout << "Writing audio buffer" << endl;
    if (!buffer || audio_stream_index < 0) return false;

    AVPacket* pkt = av_packet_alloc();
    if (!pkt) return false;

    int ret = avcodec_receive_packet(audio_codec_ctx, pkt);
    if (ret < 0) return false;

    pkt->stream_index = audio_stream_index;
    pkt->pts = av_rescale_q(audio_codec_ctx->frame_num, audio_codec_ctx->time_base, format_ctx->streams[audio_stream_index]->time_base);
    pkt->dts = pkt->pts;
    pkt->duration = av_rescale_q(1, audio_codec_ctx->time_base, format_ctx->streams[audio_stream_index]->time_base);

    ret = av_interleaved_write_frame(format_ctx, pkt);
    av_packet_unref(pkt);

    cout << "Audio buffer written" << endl;

    return ret >= 0;
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

VideoEffectEdgeDetection::VideoEffectEdgeDetection() {
}

void VideoEffectEdgeDetection::apply(const Mat& input, Mat& output, double blend) {
    Mat gray;
    cvtColor(input, gray, COLOR_BGR2GRAY);
    Mat edges;
    Canny(gray, edges, 50, 150);
    cvtColor(edges, output, COLOR_GRAY2BGR);
    output = blend * output + (1 - blend) * input;
    cout << "Edge detection applied" << endl;
}

VideoEffectMotionBlur::VideoEffectMotionBlur() {
}

void VideoEffectMotionBlur::apply(const Mat& input, Mat& output, double amount) {
    Mat kernel = Mat::zeros(5, 5, CV_32F);
    kernel.at<float>(0, 0) = 1.0 / amount;
    kernel.at<float>(1, 1) = 1.0 / amount;
    kernel.at<float>(2, 2) = 1.0 / amount;
    kernel.at<float>(3, 3) = 1.0 / amount;
    kernel.at<float>(4, 4) = 1.0 / amount;
    filter2D(input, output, -1, kernel);
    cout << "Motion blur applied" << endl;
}

VideoEffectCellAutoMata::VideoEffectCellAutoMata() {
}

void VideoEffectCellAutoMata::apply(const Mat& input, Mat& output, double neighborhood_size, double amount) {
    //find the central pixel of each neighborhood
    int center = neighborhood_size / 2;
    //create a copy of the input image
    output = input.clone();
    //loop through the image
    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {
            //get the pixel value
            Vec3b pixel = input.at<Vec3b>(y, x);
            //get the neighborhood
            int sum = 0;
            for (int i = -center; i <= center; i++) {
                for (int j = -center; j <= center; j++) {
                    //check to see if the pixel is within the bounds of the image
                    if (y + i >= 0 && y + i < input.rows && x + j >= 0 && x + j < input.cols) {
                        //get the pixel value
                        Vec3b neighbor = input.at<Vec3b>(y + i, x + j);
                        //add the pixel value to the sum
                        sum += neighbor[0] + neighbor[1] + neighbor[2];
                    }
                }
            }
            //calculate the average pixel value
            int average = sum / (neighborhood_size * neighborhood_size);
            //set the pixel value to the average
            output.at<Vec3b>(y, x) = Vec3b(average, average, average);
        }
    }
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
    MediaWriter output;
    if (!output.open("output.avi")) {
        cerr << "Error: cannot open output file" << endl;
        return 1;
    }

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

    VideoEffectBassMiddleTreble bassmidtreb;
    VideoEffectBrightness bright;
    VideoEffectEdgeDetection edgedetect;
    VideoEffectMotionBlur motionblur;
    VideoEffectCellAutoMata cellautomata;

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
            if (freq < 250) {
                bass += magnitude;
            } else if (freq < 3000) {
                middle += magnitude;
            } else {
                treble += magnitude;
            }
            brightness += magnitude;
        }

        bass /= samples_per_frame / 2;
        middle /= samples_per_frame / 2;
        treble /= samples_per_frame / 2;
        brightness /= samples_per_frame * 20;

        cout << "bass: " << bass << endl;
        cout << "middle: " << middle << endl;
        cout << "treble: " << treble << endl;
        cout << "brightness: " << brightness << endl;

        bright.apply(frame, output_frame, brightness*2);
        frame = output_frame.clone();
        edgedetect.apply(frame, output_frame, brightness);
        frame = output_frame.clone();
        motionblur.apply(frame, output_frame, brightness*12);
        frame = output_frame.clone();
        
        int neighborhood_size = brightness*1600;
        neighborhood_size = neighborhood_size % 16;
        double threshold = brightness*1200;
        cellautomata.apply(frame, output_frame, neighborhood_size, threshold);
        bassmidtreb.apply(frame, output_frame, bass*5, middle*2.5, treble*1.25);

        cout << "Writing frame to output" << endl;
        AVFrame* outputavframe = av_frame_alloc();
        outputavframe->data[0] = output_frame.data;
        outputavframe->linesize[0] = output_frame.step;
        outputavframe->width = input.getFrameWidth();
        outputavframe->height = input.getFrameHeight();
        outputavframe->format = AV_PIX_FMT_BGR24;
        outputavframe->pts = frame_index;
        output.writeVideoFrame(outputavframe);
        output.writeAudioBuffer((uint8_t*)samples.data(), samples_per_frame, sample_rate, 2);
        //free the frame
        av_frame_free(&outputavframe);
        //show the frame
        show_frame(output_frame);
        fftw_destroy_plan(plan);
        //check to see if the video has ended
        if (frame_index == frame_amount) {
            //cleanup and exit
            destroyAllWindows();
            output.~MediaWriter();
            input.~VideoInputFile();
            break;
        }

        //check to see if the user has pressed the escape key
        if (waitKey(1) == 27) {
            //cleanup and exit
            destroyAllWindows();
            output.~MediaWriter();
            input.~VideoInputFile();
            break;
        }

    }

    return 0;
}