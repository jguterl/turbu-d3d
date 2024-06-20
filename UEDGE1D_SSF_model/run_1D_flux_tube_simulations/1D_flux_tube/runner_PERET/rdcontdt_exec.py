import os
import numpy as np
import matplotlib.pyplot as plt

#execfile(os.environ['PYLIB']+"/plotmesh.py")
#execfile(os.environ['PYLIB']+"/plotcontour.py")
#execfile(os.environ['PYLIB']+"/plotr.py")
#execfile(os.environ['PYLIB']+"/plotvar.py")
#execfile(os.environ['PYLIB']+"/paws.py")

#exec(open(os.environ['PYLIB']+"/plotmesh.py").read())
#exec(open(os.environ['PYLIB']+"/plotcontour.py").read())
#exec(open(os.environ['PYLIB']+"/plotr.py").read())
#exec(open(os.environ['PYLIB']+"/plotvar.py").read())
#exec(open(os.environ['PYLIB']+"/paws.py").read())

try:
  exec(open('/fusion/projects/boundary/peretm/1D_flux_tube/runner_PERET/rdcontdt.py').read())
except SystemExit:
  print ('stop with .quit')
print ('this should print and go back to prompt')

#paws()
