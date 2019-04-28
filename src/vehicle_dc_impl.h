#pragma once

#include <vehicle_dc/vehicle_dc.h>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

class VehicleDetectorAndClassifierPyWrapper
{
public:
    struct BoundingBox
    {
        int x;
        int y;
        int width;
        int height;
    };
    struct Result
    {
        BoundingBox boundingBox;
        VehicleDCType type;
        VehicleDCColor color;
        VehicleDCOrientation orientation;
    };
    VehicleDetectorAndClassifierPyWrapper(const std::string &py_module_path,
            const std::string &model_path);
    std::vector<Result> performInference(const uint8_t *image, size_t size,
                                         int width, int height);
private:
    void setPyModulesPath(const std::string &path);
    pybind11::scoped_interpreter _interpreter;
    pybind11::object _pyVehicleDC;
    pybind11::object _pyPILImageFromBytes;
};