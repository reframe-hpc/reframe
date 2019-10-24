from paraview.simple import *
import os
basename = os.getenv('SCRATCH')
if basename is None:
    basename = "/tmp"

Version = str(GetParaViewVersion())
if(GetParaViewVersion() > 5.6):
    from paraview.modules.vtkPVClientServerCoreCorePython import (
        vtkProcessModule)
    info = GetOpenGLInformation(
        location=servermanager.vtkSMSession.RENDER_SERVER)
else:
    from paraview.servermanager import vtkProcessModule
    from vtk.vtkPVClientServerCoreRendering import vtkPVOpenGLInformation
    info = vtkPVOpenGLInformation()
    info.CopyFromObject(None)

rank = vtkProcessModule.GetProcessModule().GetPartitionId()
nbprocs = servermanager.ActiveConnection.GetNumberOfDataPartitions()

if rank == 0:
    print("ParaView Version ", Version)
    print("rank=", rank, "/", nbprocs)
    print("Vendor:   %s" % info.GetVendor())
    print("Version:  %s" % info.GetVersion())
    print("Renderer: %s" % info.GetRenderer())

Vendor = info.GetVendor().split()[0]

"""
>>> info.GetRenderer()
'SWR (LLVM 8.0, 256 bits)'
>>> info.GetVendor()
'Intel Corporation'
>>> info.GetVersion()
'3.3 (Core Profile) Mesa 18.3.3'
"""

view = GetRenderView()
view.CameraPosition = [1.642208, 1.973803, 2.14555]
view.CameraViewUp = [-0.410182, -0.492857, 0.76736]
view.CameraFocalPoint = [0.0, 0.0, 0.0]
view.OrientationAxesVisibility = 0

sphere = Sphere()
sphere.ThetaResolution = 1024
sphere.PhiResolution = 1024

pidscal = ProcessIdScalars(sphere)

rep = Show(pidscal, view)
ColorBy(rep, 'ProcessId')
processIdLUT = GetColorTransferFunction('ProcessId')
processIdLUT.AnnotationsInitialized = 1
processIdLUT.InterpretValuesAsCategories = 1

# we take colors from the pre-defined "KAAMS" found in
# ParaViewCore/ServerManager/Rendering/ColorMaps.json
IndexedColors = [
    1.0, 1.0, 1.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    1.0, 1.0, 0.0,
    1.0, 0.0, 1.0,
    0.0, 1.0, 1.0,
    0.63, 0.63, 1.0,
    0.67, 0.5, 0.33,
    1.0, 0.5, 0.75,
    0.53, 0.35, 0.7,
    1.0, 0.75, 0.5
]

a = []
for i in range(nbprocs):
    a.extend((str(i), str(i)))
processIdLUT.Annotations = a
processIdLUT.IndexedColors = IndexedColors

processIdLUTColorBar = GetScalarBar(processIdLUT, view)
processIdLUTColorBar.Title = 'PId'
processIdLUTColorBar.ComponentTitle = ''

# set color bar visibility
processIdLUTColorBar.Visibility = 1

# show color legend
rep.SetScalarBarVisibility(view, True)

view.Background = [.7, .7, .7]

Render()

view.ViewSize = [1024, 1024]
# change the pathname to a place where you have write access
filename = basename + "/coloredSphere_v" + Version + "." + Vendor + ".png"
SaveScreenshot(filename=filename, view=view)
print("writing ", filename)
