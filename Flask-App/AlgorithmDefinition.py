import cv2
import numpy
from MainFunctions import clearstraightlines, trans_normalize
from scipy.stats import pearsonr
import numpy as np

class MatchEGMAlgorithm():
    """Class for instantiating the determinsitic prediction method
    """
    
    def __init__(self):
        """ 
        Constructor for initialization
        """
        self.modelname = "MatchEGMAlgorithm"
        
    def transform(self, img, size, clearlines = True):
        """Function for transforming the image before prediction is executed. 
           Transformation includes removing dotted vertical lines, resizing, pixel intensity normalization

        Args:
            img (np array): image for transformation
            size (int,int): new size of image
            clearlines (bool, optional): if set to true the algorithm for removing dotted vertical lines is executed. Defaults to True.

        Returns:
            np array: transformed image
        """
        width = size[1]
        height = size[0]
        img = clearstraightlines(img) if clearlines else img
        img = cv2.resize(np.array(img),(width,height), cv2.INTER_NEAREST)
        img = img.astype('float64')
        img = trans_normalize(img)

        return img

    def buildseries(self, img, method):
        """Function for converting image into a time series

        Args:
            img (np array): image to be converted
            method (str): method to use for pixel intensity conversion

        Returns:
            list: list containing the image time series
        """
        series = []
        for a in img.T:
            try:
                if method == "average":
                    series.append(np.average(np.argwhere(a > 0).flatten(), weights=a[a>0]))
                elif method == "maxintensity":
                    series.append(np.average(np.argwhere(a == a.max())))
            except ZeroDivisionError:
                print("NOT Identifiable")
                series.append(0)
        return series

    def predict(self, img1, img2, method="average"):
        """Main function for prediction correlation value, given two images.

        Args:
            img1 (np array): first image, (template)
            img2 (np array)): second image (match)
            method (str, optional): method to use for pixel intensity transformation. Defaults to "average".

        Returns:
            float: correlation value rounded to two decimals
        """

        size = (min(img1.shape[0], img2.shape[0]),  min(img1.shape[1], img2.shape[1]))           
        pred = pearsonr(self.buildseries(self.transform(img1, size),method), self.buildseries(self.transform(img2, size),method))[0]
      
        return round(pred,2)