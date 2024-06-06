#include <stdio.h>
#include "xf86drm.h"

int main() {
        drmDevicePtr devices[256];
        int max_devices = drmGetDevices2(0, devices, 256);
        printf("%i\n", max_devices);
}

