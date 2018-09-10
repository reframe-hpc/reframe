
# Version = "4.4"  # tested succesfully on Fri Nov 13 15:54:52 CET 2015
Version = "5.4"    # tested succesfully on Fri Nov 13 15:54:52 CET 2015

import os
basename = os.getenv('SCRATCH')

try: paraview.simple
except: from paraview.simple import *
from vtk.vtkPVClientServerCoreRendering import vtkPVOpenGLInformation

info = vtkPVOpenGLInformation()
info.CopyFromObject(None)
print("Vendor:   %s" % info.GetVendor())
print("Version:  %s" % info.GetVersion())
print("Renderer: %s" % info.GetRenderer())
# should print
# "Vendor:   NVIDIA Corporation"
# "Version:  4.5.0 NVIDIA 375.66"
# "Renderer: Tesla P100-PCIE-16GB/PCIe/SSE2"

view = GetRenderView()

sphere = Sphere()
sphere.ThetaResolution = 2048
sphere.PhiResolution = 2048

pidscal = ProcessIdScalars(sphere)

rep = Show(pidscal)

if(GetParaViewVersion() >= 5.5):
  from vtkmodules.vtkPVClientServerCoreCorePython import vtkProcessModule
else:
  from vtkPVClientServerCoreCorePython import vtkProcessModule

print("rank=", vtkProcessModule.GetProcessModule().GetPartitionId())
print("total=", vtkProcessModule.GetProcessModule().GetNumberOfLocalPartitions())
nbprocs = servermanager.ActiveConnection.GetNumberOfDataPartitions()
drange = [0, nbprocs-1]

lt = MakeBlueToRedLT(drange[0], drange[1])
lt.NumberOfTableValues = nbprocs

rep.LookupTable = lt
rep.ColorArrayName = ("POINT_DATA", "ProcessId")

bar = CreateScalarBar(LookupTable=lt, Title="PID")
bar.TitleColor = [0, 0, 0]
bar.LabelColor = [0, 0, 0]
# bar.NumberOfLabels = 6
view.Representations.append(bar)

view.Background = [.7, .7, .7]
view.CameraViewUp = [0, 1, 0]
view.StillRender()
view.ResetCamera()
view.ViewSize = [1024, 1024]
# change the pathname to a place where you have write access
SaveScreenshot(filename=basename + "/coloredSphere_v" + Version + "_00.png",
               view=view)
# SaveScreenshot(filename = "/users/jfavre/coloredSphere_v" + Version + "_00.png", view=view)

view.CameraViewUp = [0, 0, 1]
view.CameraFocalPoint = [0, 0, 0]
view.CameraPosition = [0, 1, 0]
view.ResetCamera()
# change the pathname to a place where you have write access
SaveScreenshot(filename=basename + "/coloredSphere_v" + Version + "_01.png",
               view=view)

view.CameraViewUp = [0, 0, 1]
view.CameraFocalPoint = [0, 0, 0]
view.CameraPosition = [0, -1, 0]
view.ResetCamera()
# change the pathname to a place where you have write access
SaveScreenshot(filename=basename + "/coloredSphere_v" + Version + "_02.png",
               view=view)
