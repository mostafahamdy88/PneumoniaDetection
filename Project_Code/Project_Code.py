from PIL import Image
import numpy as np
import cv2
import glob

path = glob.glob("D:/project/Dataset_Sample/Pneumonia/*.jpeg")
print(path)

i=0
for img in path:
    '''Read Images'''
    InputImage= np.array(Image.open(img))
    
    '''1. Resizing Images and Gray Scale'''
    ResizedImage = cv2.resize(InputImage, (800,800))
    if (len(ResizedImage.shape)>2):
        ResizedImage=cv2.cvtColor(ResizedImage, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('D:/project/#Pneumonia_Resizing'+str(i+1)+'.jpeg',ResizedImage)
    
    '''2. Histogram Equalization'''
    EqualizedImage = cv2.equalizeHist(ResizedImage)
    #cv2.imwrite('D:/project/#Pneumonia_Equalized'+str(i+1)+'.jpeg',EqualizedImage) 

    '''3. Otsu Thresholded and iverse Thresholded''' 
    th_v, OtsuThresholded = cv2.threshold(EqualizedImage, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
    #cv2.imwrite('D:/project/#Pneumonia_Thresholded'+str(i+1)+'.jpeg',OtsuThresholded) 

    '''4. Morphological Operations(Opening and Closing)'''
    kernel = np.ones((7, 7),np.uint8)
    OpeningImage = cv2.morphologyEx(OtsuThresholded, cv2.MORPH_OPEN, kernel)
    ClosingImage = cv2.morphologyEx(OpeningImage, cv2.MORPH_CLOSE, kernel)
    #cv2.imwrite('D:/project/#Pneumonia_Closing'+str(i+1)+'.jpeg',ClosingImage)
    
    '''5. Remove Areas Attached To Border'''
    #Add 1 pixel white border all around
    Pad = cv2.copyMakeBorder(ClosingImage, 1,1,1,1, cv2.BORDER_CONSTANT, value=255)
    h, w = Pad.shape
    
    #Create zeros mask 2 pixels larger in each dimension
    Mask = np.zeros([h + 2, w + 2], np.uint8)
    
    #Floodfill outer white border with black (flages means connectivity)
    floodfill_Image = cv2.floodFill(Pad, Mask, (0,0), 0, (5), (0), flags=8)[1]
    
    #Remove border
    floodfill_Image = floodfill_Image[1:h-1, 1:w-1]    
    #cv2.imwrite('D:/project/#Pneumonia_Removing_Borders'+str(i+1)+'.jpeg',floodfill_Image) 

    '''6. Median Filter'''
    medianFilter = cv2.medianBlur(floodfill_Image, 21)
    OutputImage = medianFilter * ResizedImage
    #cv2.imwrite('D:/project/#Pneumonia_Median'+str(i+1)+'.jpeg',OutputImage) 

    '''7. Classification'''
    #get Number of Labels
    labelsNumber, labeledImage = cv2.connectedComponents(OutputImage, connectivity = 4)  
    #print("Number of labels:",labelsNumber,"\n","The Labeled image:",OutputImage)

    if (labelsNumber<=6):
        cv2.imwrite('D:/project/#Normal'+str(i+1)+'.jpeg',OutputImage)
    else:
        cv2.imwrite('D:/project/#Pneumonia'+str(i+1)+'.jpeg',OutputImage)
        
    i=i+1