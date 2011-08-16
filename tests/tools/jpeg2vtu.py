#!/usr/bin/env python

import sys
from vtk import *

filename = sys.argv[1]
basename = filename[:-5]

reader = vtkJPEGReader()

reader.SetFileName(basename+".jpeg")
reader.Update()

# image.Update()
image = vtkImageData()
image.DeepCopy(reader.GetOutput())

JPEGImage = image.GetPointData().GetArray("JPEGImage")

ncomponents = JPEGImage.GetNumberOfComponents()
npoints = JPEGImage.GetNumberOfTuples()
for c in range(ncomponents):
    gray = vtkDoubleArray()
    gray.SetName("JPEGImage_component%i"%c)
    gray.SetNumberOfTuples(npoints)
    gray.SetNumberOfComponents(1)
    for i in range(npoints):
        if(ncomponents==1):
            gray.SetTuple1(i, JPEGImage.GetTuple1(i))
        elif(ncomponents==3):
            data = JPEGImage.GetTuple3(i)
            gray.SetTuple1(i, data[c])
    image.GetPointData().AddArray(gray)
image.GetPointData().RemoveArray("JPEGImage")

filter = vtkDataSetTriangleFilter()
filter.SetInput(image)
filter.Update()

vtuwriter = vtkXMLUnstructuredGridWriter()
vtuwriter.SetFileName(basename+".vtu")
vtuwriter.SetInput(filter.GetOutput())
vtuwriter.Write()
