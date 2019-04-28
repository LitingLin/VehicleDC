#include "jpeg_decoder.h"

#include <string>
#include <stdexcept>
#include <limits>

JpegDecompressor::JpegDecompressor()
{
    _handle = tjInitDecompress();
    if (_handle == NULL)
    {
        throw std::runtime_error(tjGetErrorStr2(nullptr));
    }
}

JpegDecompressor::JpegDecompressor(JpegDecompressor && object)
{
    this->_handle = object._handle;
    object._handle = nullptr;
}

JpegDecompressor::~JpegDecompressor()
{
    if (_handle)
        tjDestroy(_handle);
}

void JpegDecompressor::initialize(const void *pointer, const uint64_t fileSize)
{
    if (fileSize > std::numeric_limits<unsigned long>::max())
    {
        throw std::runtime_error("Cannot process images which file size larger than " + std::to_string(std::numeric_limits<unsigned long>::max()) + " bytes.");
    }

    const auto fileSize_safeCast = static_cast<const unsigned long>(fileSize);

    if (tjDecompressHeader3(_handle, (const unsigned char*)pointer, fileSize_safeCast, &_width, &_height, &_jpegSubsamp, &_jpegColorspace) == -1)
    {
        std::string errorMessage = tjGetErrorStr2(_handle);
        throw std::runtime_error(errorMessage);
    }
    _pointer = (const unsigned char*)pointer;
    _fileSize = fileSize_safeCast;
}

int JpegDecompressor::getWidth()
{
    return _width;
}

int JpegDecompressor::getHeight()
{
    return _height;
}

int JpegDecompressor::getDecompressedSize()
{
    return _width * _height * tjPixelSize[TJPF_RGB];
}

void JpegDecompressor::decompress(void *buffer)
{
    if (tjDecompress2(_handle, _pointer, _fileSize, (unsigned char*)buffer, _width, _width * tjPixelSize[TJPF_RGB], _height, TJPF_RGB, TJFLAG_ACCURATEDCT) == -1)
    {
        std::string errorMessage = tjGetErrorStr2(_handle);
        throw std::runtime_error(errorMessage);
    }
}
