from libs.stereoMatcher cimport *
from libc.string cimport memcpy
from numpy cimport uint8_t, ndarray as ndarray_t
from numpy import ndarray, dstack, ascontiguousarray, uint8, float32, asarray
from cython cimport boundscheck, wraparound

cdef class PyStereoMatcher:
    cdef:
        Mat _disparityMap
        vector[Mat] _images

    def __cinit__(self, uchar[:,:,:,:]&leftImgs, uchar[:,:,:,:]&rightImgs, int ndisp, const string methods, const string size):
        try:
            self.mainLoop(leftImgs, rightImgs, ndisp, methods, size)
        except Exception as e:
            print("[Info Error]:", e)

    @boundscheck(False)
    @wraparound(False)
    cdef void mainLoop(self, uchar[:,:,:,:]&leftImgs, uchar[:,:,:,:]&rightImgs, int ndisp, const string methods, const string size):
        cdef:
            Mat LFrame, RFrame
            int index = 0
            int lsize, rsize
        lsize = leftImgs.shape[0]     
        rsize = rightImgs.shape[0]
        while True:

            LFrame = self.np2Mat3D(leftImgs[index])
            RFrame = self.np2Mat3D(rightImgs[index])

            assert LFrame.rows == RFrame.rows and LFrame.cols == RFrame.cols, "the amount of rows or cols ist not identical"

            print("start matching..." )
            print("frame number -> ", index)

            self._disparityMap.release()
            self._disparityMap = self.calcDisparityMaps(LFrame, RFrame, ndisp, methods)

            self._disparityMap *= 65535 / ndisp;
            self._images.push_back(self._disparityMap.clone());

            index +=1

            if lsize == index and rsize == index:
                break 

    @boundscheck(False)
    @wraparound(False)
    cdef ndarray_t[uint8_t, ndim=4] _disparityMaps(self, vector[Mat]&images):
        cdef: 
            int i
        
        output = []

        for i in range(images.size()):
            img = self.Mat2np(images[i])
            output.append(img[:])
        
        return asarray(output, dtype=uint8)


    @boundscheck(False)
    @wraparound(False)
    cdef inline Mat calcDisparityMaps(self, Mat&leftImg, Mat&rightImg, int ndisp, const string methods):
        cdef:
            BP * BPmatcher
            MBP * MBPmatcher
            SAD * SADmatcher
            Mat disparity
       
        if methods == "BP" or methods == "bp":
            BPmatcher = new BP(leftImg, rightImg, ndisp, 1, 2 * float(ndisp), 5)
            try:
                disparity = BPmatcher.do_match()
            finally:
                del BPmatcher
        elif methods == "MBP" or methods == "mbp":
            MBPmatcher = new MBP(leftImg, rightImg, ndisp, 1, 2*float(ndisp), 5)
            try:
                disparity = MBPmatcher.do_match()
            finally:
                del MBPmatcher
        else:
            SADmatcher = new SAD(2, ndisp)
            try:
                disparity = SADmatcher.do_match(leftImg, rightImg)
            finally:
                del SADmatcher
        return disparity
    
    @boundscheck(False)
    @wraparound(False)
    cdef inline Mat np2Mat3D(self, uchar[:,:,:] ary):
        assert ary.ndim==3 and ary.shape[2]==3, "ASSERT::3channel RGB only!!"
        ary = dstack((ary[...,2], ary[...,1], ary[...,0])) #RGB -> BGR

        cdef ndarray_t[uint8_t, ndim=3, mode ='c'] np_buff = ascontiguousarray(ary, dtype=uint8)
        cdef unsigned int* im_buff = <unsigned int*> np_buff.data
        cdef int r = ary.shape[0]
        cdef int c = ary.shape[1]
        cdef Mat m
        m.create(r, c, CV_8UC3)
        memcpy(m.data, im_buff, r*c*3)
        return m


    @boundscheck(False)
    @wraparound(False)
    cdef inline object Mat2np(self, Mat m):
        # Create buffer to transfer data from m.data
        cdef Py_buffer buf_info

        # Define the size / len of data
        cdef size_t len = m.rows*m.cols*m.elemSize() # m.channels()*sizeof(CV_8UC3)

        # Fill buffer
        PyBuffer_FillInfo(&buf_info, NULL, m.data, len, 1, PyBUF_FULL_RO)

        # Get Pyobject from buffer data
        Pydata  = PyMemoryView_FromBuffer(&buf_info)

        # Create ndarray with data
        # the dimension of the output array is 2 if the image is grayscale
        if m.channels() >1 :
            shape_array = (m.rows, m.cols, m.channels())
        else:
            shape_array = (m.rows, m.cols)

        if m.depth() == CV_32F :
            ary = ndarray(shape=shape_array, buffer=Pydata, order='c', dtype=float32)
        else :
            #8-bit image
            ary = ndarray(shape=shape_array, buffer=Pydata, order='c', dtype=uint8)

        if m.channels() == 3:
            # BGR -> RGB
            ary = dstack((ary[...,2], ary[...,1], ary[...,0]))

        # Convert to numpy array
        pyarr = asarray(ary)
        return pyarr

    def disparity_maps(self):
        return self._disparityMaps(self._images)

       
    
