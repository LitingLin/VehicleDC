#include <vehicle_dc/vehicle_dc.h>
#include "vehicle_dc_impl.h"

#include <cstdlib>
#include <exception>
#include <string>
#include <memory>

#include <config.h>

struct Context {
    std::string errorString;
    VehicleDetectorAndClassifierPyWrapper *ptr;
    std::string pyModulePath = VEHICLE_DC_PY_MODULE_PATH;
    std::string modelPath = VEHICLE_DC_MODEL_PATH;
    std::vector<VehicleDetectorAndClassifierPyWrapper::Result> results;
} g_context;

void VehicleDCConfig(const char *py_module_path, const char *model_path) {
    g_context.pyModulePath = py_module_path;
    g_context.modelPath = model_path;
}

#define BEGIN_EXCEPTION_SAFE_EXECUTION \
try {
#define END_EXCEPTION_SAFE_EXECUTION \
} catch (std::exception &e) { \
    g_context.errorString = e.what(); \
    return -1; \
} catch (...) { \
    g_context.errorString = "Unknown internal error"; \
    return -1; }

int VehicleDCInitialize() {
    BEGIN_EXCEPTION_SAFE_EXECUTION
        if (g_context.ptr) delete g_context.ptr;
        g_context.ptr = nullptr;
        g_context.ptr = new VehicleDetectorAndClassifierPyWrapper(g_context.pyModulePath, g_context.modelPath);
        return 0;
    END_EXCEPTION_SAFE_EXECUTION
}

const char *VehicleDCVehicleTypeToString(VehicleDCType type) {
    switch (type) {
        case PassengerCar:
            return "PassengerCar";
            break;
        case SaloonCar:
            return "SaloonCar";
            break;
        case ShopTruck:
            return "ShopTruck";
            break;
        case Suv:
            return "Suv";
            break;
        case Trailer:
            return "Trailer";
            break;
        case Truck:
            return "Truck";
            break;
        case Van:
            return "Van";
            break;
        case Waggon:
            return "Waggon";
            break;
        default:
            abort();
    }
}

const char *VehicleDCVehicleColorToString(VehicleDCColor type) {
    switch (type) {
        case Black:
            return "Black";
            break;
        case Blue:
            return "Blue";
            break;
        case Brown:
            return "Brown";
            break;
        case Gray:
            return "Gray";
            break;
        case Green:
            return "Green";
            break;
        case Pink:
            return "Pink";
            break;
        case Red:
            return "Red";
            break;
        case White:
            return "White";
            break;
        case Yellow:
            return "Yellow";
            break;
        default:
            abort();
    }
}

const char *VehicleDCVehicleOrientationToString(VehicleDCOrientation type) {
    switch (type) {
        case Front:
            return "Front";
            break;
        case Rear:
            return "Rear";
            break;
        default:
            abort();
    }
}

int VehicleDCPerformInference(const uint8_t *image, size_t size, int width, int height, int *number_of_vehicles_detected) {
    BEGIN_EXCEPTION_SAFE_EXECUTION
        g_context.results = g_context.ptr->performInference(image, size, width, height);
        *number_of_vehicles_detected = static_cast<int>(g_context.results.size());
        return 0;
    END_EXCEPTION_SAFE_EXECUTION
}

int VehicleDCGetResult(int index, int *x, int *y, int *width, int *height,
        VehicleDCType *vehicle_type,
        VehicleDCColor *vehicle_color,
        VehicleDCOrientation *vehicle_orientation) {
    auto &result = g_context.results[index];
    *x = result.boundingBox.x;
    *y = result.boundingBox.y;
    *width = result.boundingBox.width;
    *height = result.boundingBox.height;
    *vehicle_type = result.type;
    *vehicle_color = result.color;
    *vehicle_orientation = result.orientation;
    return 0;
}

void VehicleDCRelease() {
    try {
        if (g_context.ptr) delete g_context.ptr;
    } catch (...) {}
    g_context.ptr = nullptr;
    g_context.results.resize(0);
    g_context.errorString.resize(0);

}

const char *VehicleDCGetLastErrorString() {
    return g_context.errorString.c_str();
}
