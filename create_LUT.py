from PIL import Image    # Python Imaging Library
import math                # Maths functions
import sys                # Allows us to access function args
import os                # Allows us to split the text for saving the file
import numpy as np
# import matplotlib.pyplot as plt
# import multiprocessing
# import parmap
import time

class LookUpTable():
    def __init__(self, Wcube, iinterp, imode):
        self.imode = imode
        self.Hcube = Wcube
        self.Wcube = Wcube
        self.Hequi = Wcube
        if self.imode == 0:
            self.Wequi = Wcube
        elif self.imode == 1:
            self.Wequi = int(Wcube*2)
        self.lookup_table = np.zeros((self.Hequi, self.Wequi, 3)) - 1  # -1 to get error if table not correctly filled
        self.iinterp = iinterp

    def unit3DToUnit2D(self, x, y, z, faceIndex):

        if(faceIndex == "X+"):
            x2D = y+0.5
            y2D = z+0.5
        elif(faceIndex == "Y+"):
            x2D = (x*-1)+0.5
            y2D = z+0.5
        elif(faceIndex == "X-"):
            x2D = (y*-1)+0.5
            y2D = z+0.5
        elif(faceIndex == "Y-"):
            x2D = x+0.5
            y2D = z+0.5
        elif(faceIndex == "Z+"):
            x2D = y+0.5
            y2D = (x*-1)+0.5
        else:
            x2D = y+0.5
            y2D = x+0.5

        y2D = 1-y2D

        return (x2D, y2D)

    def projectX(self, theta, phi, sign):
        x = sign*0.5
        faceIndex = "X+" if sign == 1 else "X-"
        rho = float(x)/(math.cos(theta)*math.sin(phi))
        y = rho*math.sin(theta)*math.sin(phi)
        z = rho*math.cos(phi)
        return (x, y, z, faceIndex)

    def projectY(self, theta, phi, sign):
        y = sign*0.5
        faceIndex = "Y+" if sign == 1 else "Y-"
        rho = float(y)/(math.sin(theta)*math.sin(phi))
        x = rho*math.cos(theta)*math.sin(phi)
        z = rho*math.cos(phi)
        return (x, y, z, faceIndex)

    def projectZ(self, theta, phi, sign):
        z = sign*0.5
        faceIndex = "Z+" if sign == 1 else "Z-"
        rho = float(z)/math.cos(phi)
        x = rho*math.cos(theta)*math.sin(phi)
        y = rho*math.sin(theta)*math.sin(phi)
        return (x, y, z, faceIndex)

    def getLookUpTable(self, x, y, index):
        # print(x,y)
        if(index == "X+"):
            return [0, x, y]
        elif(index == "X-"):
            return [1, x, y]
        elif(index == "Y+"):
            return [2, x, y]
        elif(index == "Y-"):
            return [3, x, y]
        elif(index == "Z+"):
            return [4, x, y]
        elif(index == "Z-"):
            return [5, x, y]

    def convertEquirectUVtoUnit2D(self, theta, phi, squareLength):

        # calculate the unit vector

        x = math.cos(theta)*math.sin(phi)
        y = math.sin(theta)*math.sin(phi)
        z = math.cos(phi)

        # find the maximum value in the unit vector

        maximum = max(abs(x), abs(y), abs(z))
        xx = x/maximum
        yy = y/maximum
        zz = z/maximum

        # project ray to cube surface

        if(xx == 1 or xx == -1):
            (x, y, z, faceIndex) = self.projectX(theta, phi, xx)
        elif(yy == 1 or yy == -1):
            (x, y, z, faceIndex) = self.projectY(theta, phi, yy)
        else:
            (x, y, z, faceIndex) = self.projectZ(theta, phi, zz)

        (x, y) = self.unit3DToUnit2D(x, y, z, faceIndex)

        x *= squareLength
        y *= squareLength

        if self.iinterp == "nearest":
            x = int(x)
            y = int(y)

        return {"index": faceIndex, "x": x, "y": y}

    def create_table(self, save_dir):
        # if self.Wequi != self.Wcube:
        #     print("Warning: not same width for cube and equi!!!")

        for loopX in range(0, self.Wequi):
            for loopY in range(0, self.Hequi):
                U = float(loopX)/(self.Wequi-1)
                V = float(loopY)/(self.Hequi-1)
                theta = U*2*math.pi
                phi = V*math.pi
                cart = self.convertEquirectUVtoUnit2D(theta, phi, self.Hequi)

                self.lookup_table[loopY, loopX, :] = self.getLookUpTable(
                    cart["x"], cart["y"], cart["index"])
                # print(loopX,loopY,self.lookup_table[loopY, loopX, :],cart)

        print(self.lookup_table.shape)
        np.save(os.path.join(save_dir, 'Lookup_Table_' +
                             str(self.Wequi)+'_'+str(self.Hequi)+'_'+str(self.iinterp)+'_'+str(self.imode)+'.npy'), self.lookup_table)

def create_lookup_table(Wcube, save_dir, iinterp, imode):
    LUT = LookUpTable(Wcube, iinterp, imode)
    print("Creating LookUp Table Cube "+str(LUT.Wcube)+"x"+str(LUT.Hcube) +
          " to Equi "+str(LUT.Wequi)+"x"+str(LUT.Hequi)+" imode "+str(imode))

    LUT.create_table(save_dir)
    # print(LUT.lookup_table)


