#pragma once

#ifdef _WIN32
#ifdef BUILDING_SHARED_LIB
#define VEHICLE_DC_EXPORTS __declspec(dllexport)
#else
#define VEHICLE_DC_EXPORTS __declspec(dllimport)
#endif
#else
#ifdef BUILDING_SHARED_LIB
#define VEHICLE_DC_EXPORTS __attribute__((visibility("default")))
#else
#define VEHICLE_DC_EXPORTS
#endif
#endif

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif
enum VehicleDCType {
    PassengerCar,
    SaloonCar,
    ShopTruck,
    Suv,
    Trailer,
    Truck,
    Van,
    Waggon
};
enum VehicleDCColor {
    Black,
    Blue,
    Brown,
    Gray,
    Green,
    Pink,
    Red,
    White,
    Yellow
};
enum VehicleDCOrientation {
    Front,
    Rear
};

VEHICLE_DC_EXPORTS
void VehicleDCConfig(const char *py_module_path, const char *model_path);

VEHICLE_DC_EXPORTS
int VehicleDCInitialize();

// Must be RGB
VEHICLE_DC_EXPORTS
int VehicleDCPerformInference(const uint8_t* image, size_t size,
        int width, int height,
        int* number_of_vehicles_detected);

VEHICLE_DC_EXPORTS
int VehicleDCGetResult(int index,
        int* x, int* y, int* width, int* height,
        VehicleDCType* vehicle_type,
        VehicleDCColor *vehicle_color,
        VehicleDCOrientation* vehicle_orientation);

VEHICLE_DC_EXPORTS
void VehicleDCRelease();

VEHICLE_DC_EXPORTS
const char* VehicleDCVehicleTypeToString(VehicleDCType type);

VEHICLE_DC_EXPORTS
const char* VehicleDCVehicleColorToString(VehicleDCColor type);

VEHICLE_DC_EXPORTS
const char* VehicleDCVehicleOrientationToString(VehicleDCOrientation type);

VEHICLE_DC_EXPORTS
const char* VehicleDCGetLastErrorString();

#ifdef __cplusplus
}
#endif