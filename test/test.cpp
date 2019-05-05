#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include <test.h>
#include <vehicle_dc/vehicle_dc.h>

#include <stdio.h>
#include "jpeg_decoder.h"

std::vector<unsigned char> fileReadAll(const std::string &path) {
    FILE *f = fopen(path.c_str(), "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::vector<unsigned char> buffer;
    buffer.resize(fsize);

    fread(buffer.data(), 1, fsize, f);
    fclose(f);
    return buffer;
}

struct BoundingBox {
    int x;
    int y;
    int width;
    int height;
};
struct Result {
    BoundingBox boundingBox;
    VehicleDCType type;
    VehicleDCColor color;
    VehicleDCOrientation orientation;
};

bool operator==(const Result &l, const Result &r) {
    return memcmp(&l, &r, sizeof(Result)) == 0;
}

std::ostream &operator<<(std::ostream &stream, Result &value) {
    stream << "Bounding Box: [x: " << value.boundingBox.x << ", y: " << value.boundingBox.y
           << ", width: " << value.boundingBox.width << ", height: " << value.boundingBox.height << "]. "
           << "Vehicle Type: " << VehicleDCVehicleTypeToString(value.type) << ". "
           << "Vehicle Color: " << VehicleDCVehicleColorToString(value.color) << ". "
           << "Vehicle Orientation: " << VehicleDCVehicleOrientationToString(value.orientation) << ".";
    return stream;
}

TEST_CASE("Common", "[common]") {
    std::vector<unsigned char> jpegImage = fileReadAll(std::string(TESTING_SOURCE_PATH) + "/0001.jpg");
    JpegDecompressor jpegDecompressor;
    jpegDecompressor.initialize(jpegImage.data(), jpegImage.size());
    auto imageSize = jpegDecompressor.getDecompressedSize();
    std::vector<unsigned char> buffer;
    buffer.resize(imageSize);
    jpegDecompressor.decompress((void *) buffer.data());

    VehicleDCConfig(TESTING_PY_MODULE_PATH, TESTING_MODEL_PATH);
    if (VehicleDCInitialize() != 0) throw std::runtime_error(VehicleDCGetLastErrorString());
    int numberOfVehicleDetected = 0;
    if (VehicleDCPerformInference(buffer.data(), buffer.size(),
                                  jpegDecompressor.getWidth(), jpegDecompressor.getHeight(),
                                  &numberOfVehicleDetected) != 0)
        throw std::runtime_error(VehicleDCGetLastErrorString());
    REQUIRE(numberOfVehicleDetected == 16);
    std::vector<Result> results;
    results.reserve(numberOfVehicleDetected);
    std::cout << "Predicted: " << numberOfVehicleDetected << " vehicle(s)." << std::endl;
    for (int i = 0; i < numberOfVehicleDetected; ++i) {
        Result result;
        VehicleDCGetResult(i, &result.boundingBox.x, &result.boundingBox.y,
                           &result.boundingBox.width, &result.boundingBox.height,
                           &result.type, &result.color, &result.orientation);
        results.push_back(result);
        std::cout << "#" << i + 1 << ": " << result << std::endl;
    }

    VehicleDCRelease();

    std::vector<Result> expectedValues = {
            {{276,  545, 169, 196},
                    SaloonCar,
                    Red,
                    Rear},
            {{56,   878, 262, 179},
                    SaloonCar,
                    White,
                    Front},
            {{741,  324, 95,  94},
                    SaloonCar,
                    White,
                    Front},
            {{571,  379, 101, 124},
                    Van,
                    Black,
                    Rear},
            {{512,  530, 161, 210},
                    SaloonCar,
                    White,
                    Front},
            {{876,  384, 142, 120},
                    SaloonCar,
                    White,
                    Front},
            {{989,  635, 192, 220},
                    SaloonCar,
                    White,
                    Front},
            {{805,  259, 77,  66},
                    SaloonCar,
                    White,
                    Front},
            {{603,  290, 96,  111},
                    SaloonCar,
                    Green,
                    Front},
            {{397,  396, 118, 136},
                    SaloonCar,
                    Black,
                    Front},
            {{736,  421, 120, 128},
                    SaloonCar,
                    White,
                    Front},
            {{460,  301, 112, 105},
                    SaloonCar,
                    White,
                    Front},
            {{1368, 474, 194, 171},
                    SaloonCar,
                    White,
                    Rear},
            {{833,  312, 98,  96},
                    SaloonCar,
                    White,
                    Rear},
            {{774,  242, 83,  68},
                    SaloonCar,
                    Green,
                    Front},
            {{694,  205, 77,  67},
                    SaloonCar,
                    Black,
                    Rear}
    };
    REQUIRE(expectedValues == results);
}