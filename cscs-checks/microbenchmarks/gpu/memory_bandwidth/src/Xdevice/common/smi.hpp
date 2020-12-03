#ifndef __DEFINED_COMMON_SMI__
#define __DEFINED_COMMON_SMI__

#include <iostream>

class Smi
{
private:
  /*
   * Indicator that shows if the Smi class has any active instances.
   *
   * This is set to 1 if the activeSmiInstances is greater that 0.
   *
   */
  static int smiIsActive;

  /*
   * Active instance counter.
   *
   * This counter is incrementer/decremented in the class constructor/destructor.
   *
   */
  static int activeSmiInstances;

  /*
   * Number of present devices in the current node.
   *
   * This gets set during the class construction.
   *
   */
  unsigned int numberOfDevices;

public:
  Smi();
  ~Smi();

  /*
   * Gets the number of devices.
   *
   * Parameters:
   *   + n: address of the integer where to return the number of devices.
   */
  void getNumberOfDevices(int * n) const;

  /*
   * Test wheter the requested GPU id is within the range of available IDs in the curent node.
   *
   * This function can exit the program if the requested value is outside the valid range.
   *
   * Parameters:
   *   + id: Device ID.
   */
  void checkGpuIdIsSensible(int id) const;

  /*
   * Pin the host thread to a CPU with affinity to a given device.
   *
   * Parameters:
   *   + id: Device ID.
   */
  void setCpuAffinity(int id);

  /*
   * Get the current temperature of a given device.
   *
   * Parameters:
   *   + id: Device ID.
   */
  void getGpuTemp(int id, float * temp) const;

  /*
   * Get the total memory size present in a device.
   *
   * Parameters:
   *   + id: Device id.
   *   + mSize: Size (in bytes) of the total memory.
   *
   */
  void getDeviceMemorySize(int id, size_t * mSize) const;

  /*
   * Get the available memory size in a given device.
   *
   * Parameters:
   *   + id: Device id.
   *   + mSize: Size (in bytes) of the available memory.
   *
   */
  void getDeviceAvailMemorySize(int id, size_t * mSize) const;
};

/*
 * Default values for the static members.
 */
int Smi::smiIsActive = 0;
int Smi::activeSmiInstances = 0;


/*
 * Member functions with a common implementation across platforms.
 */

void Smi::getNumberOfDevices(int * n) const
{
  *n = this->numberOfDevices;
}

void Smi::checkGpuIdIsSensible(int id) const
{
  if (id < 0 || id >= numberOfDevices)
  {
    std::cerr << "Requested device ID is out of range from the existing devices." << std::endl;
    exit(1);
  }
}

#endif
