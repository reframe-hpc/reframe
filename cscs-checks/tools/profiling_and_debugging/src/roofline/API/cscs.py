# Taken from /opt/intel/advisor_2018/pythonapi/examples/roofline.py
#
# The roofline model is based on GFLOPS and Arithmetic Intensity (AI):
#   "Self GFLOPS" = "Self GFLOP" / "Self Elapsed Time"
#   "Self GB/s" = "Self Memory GB" / "Self Elapsed Time"
#   "Self AI" = "Self GFLOPS" / "Self GB/s"
import sys
try:
    import advisor
except ImportError:
    print('''Import error: Python could not load advisor python library.
    Possible reasons:\n 1. Python cannot resolve path to Advisor\'s pythonapi
directory. To fix, either manually add path to the pythonapi directory into
PYTHONPATH environment variable, or use advixe-vars.* scripts to set up
product environment variables automatically.\n 2. Incompatible runtime
versions used by advisor python library and other packages (such as
matplotlib or pandas). To fix, either try to change import order or update
other package version if possible. 3. cscs: try
sys.path.append(\'/opt/intel/advisor/pythonapi\')''')
    sys.exit(1)

if len(sys.argv) < 2:
    print('Usage: "python {} path_to_project_dir"'.format(__file__))
    sys.exit(2)

project = advisor.open_project(sys.argv[1])
data = project.load(advisor.SURVEY)
# data = project.load(advisor.ALL)
rows = [{col: row[col] for col in row} for row in data.bottomup]

# --- Extract values from the report and compute our arithmetic_intensity:
self_elapsed_time = float(rows[0]['self_elapsed_time'])

self_memory_gb = float(rows[0]['self_memory_gb'])
self_gb_s = float(rows[0]['self_gb_s'])
_self_gb_s = self_memory_gb / self_elapsed_time

self_gflop = float(rows[0]['self_gflop'])
self_gflops = float(rows[0]['self_gflops'])
_self_gflops = self_gflop / self_elapsed_time

self_arithmetic_intensity = float(rows[0]['self_arithmetic_intensity'])
_self_arithmetic_intensity = _self_gflops / _self_gb_s

# --- Reported values:
print('self_elapsed_time', self_elapsed_time)
print('self_memory_gb', self_memory_gb)
print('self_gb_s', self_gb_s)
print('self_gflop', self_gflop)
print('self_gflops', self_gflops)
print('self_arithmetic_intensity', self_arithmetic_intensity)

print('_self_gb_s', _self_gb_s, self_gb_s)
print('_self_gflops', _self_gflops, self_gflops)
print('_self_arithmetic_intensity', _self_arithmetic_intensity,
      self_arithmetic_intensity)

print('gap _self_gb_s', _self_gb_s-self_gb_s)
print('gap _self_gflops', _self_gflops-self_gflops)
print('gap _self_arithmetic_intensity',
      _self_arithmetic_intensity-self_arithmetic_intensity)

# --- Compare the roofline report:
print('returned AI gap = {:.16f}'.
      format(_self_arithmetic_intensity-self_arithmetic_intensity))
print('returned GFLOPS gap = {:.16f}'.
      format(_self_gflops-self_gflops))
