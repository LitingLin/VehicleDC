#include "vehicle_dc_impl.h"

#include "config.h"

VehicleDetectorAndClassifierPyWrapper::VehicleDetectorAndClassifierPyWrapper(
        const std::string &py_module_path, const std::string &model_path) {
    try {
        setPyModulesPath(py_module_path);
        _pyVehicleDC = pybind11::module::import("inference").attr("VehicleDC")(model_path);
        _pyPILImageFromBytes = pybind11::module::import("PIL").attr("Image").attr("frombytes");
    } catch (std::exception &e) {
        throw std::runtime_error(e.what());
    } catch (...) {
        throw std::runtime_error("Unknown internal error");
    }
}

void VehicleDetectorAndClassifierPyWrapper::setPyModulesPath(const std::string &path) {
    auto os_module = pybind11::module::import("sys");
    auto path_attr = os_module.attr("path");
    path_attr.attr("append")(path);
}

std::vector<VehicleDetectorAndClassifierPyWrapper::Result>
VehicleDetectorAndClassifierPyWrapper::performInference(const uint8_t *image, size_t size,
        int width, int height) {
    auto pyImage = _pyPILImageFromBytes("RGB",
            pybind11::make_tuple(width, height),
            pybind11::bytes(reinterpret_cast<const char*>(image), size));
    auto pyResult = _pyVehicleDC.attr("detect_classify")(pyImage).cast<pybind11::tuple>();
    auto pyBoundingBoxes = pyResult[0].cast<pybind11::list>();
    auto pyVehicleColors = pyResult[1].cast<pybind11::list>();
    auto pyVehicleOrientations = pyResult[2].cast<pybind11::list>();
    auto pyVehicleTypes = pyResult[3].cast<pybind11::list>();
    std::vector<VehicleDetectorAndClassifierPyWrapper::Result> result;
    auto resultSize = pyBoundingBoxes.size();
    result.reserve(resultSize);
    for (size_t index = 0; index < resultSize; ++index) {
        auto pyBoundingBox = pyBoundingBoxes[index].cast<pybind11::tuple>();
        auto vehicleColor = pyVehicleColors[index].cast<int>();
        auto vehicleOrientation = pyVehicleOrientations[index].cast<int>();
        auto vehicleType = pyVehicleTypes[index].cast<int>();
        result.push_back(Result{
                BoundingBox{
                        pyBoundingBox[0].cast<int>(),
                        pyBoundingBox[1].cast<int>(),
                        pyBoundingBox[2].cast<int>(),
                        pyBoundingBox[3].cast<int>()},
                static_cast<VehicleDCType>(vehicleType),
                static_cast<VehicleDCColor>(vehicleColor),
                static_cast<VehicleDCOrientation>(vehicleOrientation)
        });
    }
    return result;
}
