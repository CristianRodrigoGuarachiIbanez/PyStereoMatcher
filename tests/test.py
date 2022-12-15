

from PyImageCropper.pyCropper import PyCropper
from PyStereoMatching.PyStereoMatcher import PyStereoMatcher
from zipfile import ZipFile
import sys, cv2 as cv
import os, shutil
import numpy as np
import matplotlib.pyplot as plt 
import h5py as h5

class FileManager:
    def __init__(self, fname) -> None:
        self._dirFiles = fname
        self._zipFiles = list()
        self._lImages = np.empty(0)
        self._rImages = np.empty(0)
        self._rdisparity_maps = np.empty(0)
        self._ldisparity_maps = np.empty(0)
    
    def _reset(self, dir_path, type, ext ):
        self._cropper = PyCropper(dir_path, type, ext)

    def _collectImages(self, dir, start, end, c, p):
        imgCropper = PyCropper(dir, b".png", 4)
        return imgCropper.crop_images(start, end, ord(c), ord(p)) 

    @staticmethod
    def _writer(path, group, ds1, data1, ds2=None, data2=None):
        with h5.File(path, "w") as container:
            g=container.create_group(group)
            g.create_dataset(ds1, data=data1)
            if ds2:
                g.create_dataset(ds2, data=data2)
            else:
                pass
        h5.close()

    @staticmethod
    def _append(arr1, arr2):
        return np.append(arr1, arr2, axis=0)
    
    def _zipfile(self, path, fname, start, end):
        self._reset(dir_path=path, type=b".zip", ext=4)
        _zipFiles = self._cropper.directories()
        for i in range(len(_zipFiles)):
            sufix = str(_zipFiles[i]).split("_")[0]
            with ZipFile(_zipFiles[i].decode("utf-8"), "r") as archive:
                for filename in archive.namelist():
                    file = filename.split("/")
                    if file[0] == fname:
                        archive.extract(filename)
                    else:
                        pass
            path = os.getcwd() + "/" + fname
            if os.path.exists(path):
                lImgs = self._collectImages(path.encode("utf-8"), start, end, "l", "n")
                rImgs = self._collectImages(path.encode("utf-8"), start, end, "r", "n")
                shutil.rmtree(path)
                # if self._lImages.size != 0:
                #     self._lImages = self._append(self._lImages, lImgs)                    
                # else:
                #     self._lImages = lImgs[:]
                
                # if self._rImages.size != 0:
                #     self._rImages = self._append(self._rImages, rImgs)   
                # else:
                #     self._rImages = rImgs[:]
                 
                matcher = PyStereoMatcher(np.asarray(lImgs, dtype=np.uint8), np.asarray(rImgs, dtype=np.uint8), 64, b"MBP", b"big")

                if sufix == "re" or sufix == "ra":
                    if self._rdisparity_maps.size != 0:
                        self._rdisparity_maps = self._append(self._rdisparity_maps, matcher.disparity_maps())
                    else:
                        self._rdisparity_maps = matcher.disparity_maps()

                else:
                    if self._ldisparity_maps.size != 0:
                        self._ldisparity_maps = self._append(self._ldisparity_maps, matcher.disparity_maps())
                    else:
                        self._ldisparity_maps = matcher.disparity_maps()   
            
            else:
                raise Exception("the directory {} could not be exported successfully!".format(path))

            # if i == 2:
            #     self._writer("./dp.h5", "disparity_maps", "rImages", self._rdisparity_maps, "lImages", self._ldisparity_maps)
            #     sys.exit()
        self._writer(os.getcwd() + "/dp.h5", "images", "disparity_maps", self._append(self._ldisparity_maps, self._rdisparity_maps))
    

    def _show_images(self, howMany):
       self._cropper.show_original_images(2)  

    def query(self, dir_path=b"../../../image_outputs/", type = b".png", ext =4):
        self._reset(dir_path, type, ext)
    
    def zipQuery(self, path, start=[120, 386], end=[548, 900]):
        self._zipfile(path, self._dirFiles, start, end)

    def directories(self):
        return self._cropper.directories()
    
    def left_images(self):
        return self._lImages
    
    def right_images(self):
        return self._rImages
    
    def write(self, path, group, dataset):

        if group == "left_images":
            self._writer(path, group, dataset, self._lImages)
        else:
            self._writer(path, group, dataset, self._rImages)


# l = 108 | L = 76
# r = 114 | R = 82
# n = 110
# j = 106 | J = 74



if __name__ == "__main__":

     pass