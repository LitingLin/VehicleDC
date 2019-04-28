#pragma once

#include <stdint.h>
#include <turbojpeg.h>

class JpegDecompressor
{
public:
    JpegDecompressor();
    JpegDecompressor(const JpegDecompressor &) = delete;
    JpegDecompressor(JpegDecompressor && object);
    ~JpegDecompressor();
    void initialize(const void *pointer, const uint64_t fileSize);
    int getWidth();
    int getHeight();
    int getDecompressedSize();
    void decompress(void *buffer);
private:
    const unsigned char *_pointer;
    unsigned long _fileSize;
    tjhandle _handle;
    int _width, _height;
    int _jpegSubsamp, _jpegColorspace;
};