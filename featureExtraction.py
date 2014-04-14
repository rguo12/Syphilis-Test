############################################################################
#                                                                          #
#Sample Detector v0.1 By Guo Ruocheng: mcspinemo@gmail.com                 #
#3 basic features and 2 optional features included                         #
#                                                                          #
#Please run it with python 2.7                                             #
#Please modify the trainpath,testpath, myid and mypath before execute      #
#Manual decision boundary included for generate target value of train data #
#And for generating test data without target value                         #
#For predicting target value of test data, use classifier.py               #
#Result would be recorded in a csv file                                    #
#                                                                          #
############################################################################


import cv2
import numpy as np
import csv



class circleContour:
    
    def __init__(self,contour,centre,radius):
        self.contour = contour
        self.centre = centre
        self.radius = radius
    

class detector:
    
    def __init__(self,myid):
        self.myid = myid
    
    #calculate histogram, just for test
    def calcAndDrawHist(self,img, color):
        hist = cv2.calcHist([img], [0], None, [256], [0.0, 255.0])
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
        #Even we only want maxVal, this command would give us more, we have to use some vars to receive 'em
        histImg = np.zeros([256, 256, 3], np.uint8)
        hpt = int(0.9* 256)
        
        for h in range(256):
            intensity = int(hist[h]*hpt/maxVal)
            cv2.line(histImg, (h, 256), (h,256 - intensity), color)
        
        return histImg
    
    
    
    def drawContoursWithReasonableRadiusByCircle(self,threshToDo, icircle):
        #find contours
        contoursABS_bar, hierachyABS_bar = cv2.findContours(threshToDo,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        #Two conditions to recognize images with sample from ones without sample
        #Use number of contours to do this, ones without sample have much more contours
        print len(contoursABS_bar)
        if len(contoursABS_bar)>=2 and len(contoursABS_bar)<20000:
            circleContoursList = []
            
            #Iterate through all contours
            for indice in range(0,len(contoursABS_bar)):
                
                #Use the area of contour to find out the contours can represent the boundary of the sample
                if cv2.contourArea(contoursABS_bar[indice]) > 100000.0 and cv2.contourArea(contoursABS_bar[indice]) < 10000000.0:
                    
                    #Simplify the contours and use circles to represent them
                    #Save these circles to circleList and return it
                    myCircleContour = circleContour([],(0,0),0)
                    myCircleContour.contour = cv2.approxPolyDP(contoursABS_bar[indice], 0.01*cv2.arcLength(contoursABS_bar[indice],True), True)
                    
                    (x_tmp,y_tmp), circleContourRadius_tmp = cv2.minEnclosingCircle(myCircleContour.contour)
               
                    myCircleContour.centre = (int(x_tmp),int(y_tmp))
                    myCircleContour.radius = int(circleContourRadius_tmp)
                    circleContoursList.append(myCircleContour)
                    cv2.circle(icircle, myCircleContour.centre, myCircleContour.radius,(255,0,0),2)
            
            print len(circleContoursList)
            
            return circleContoursList
    
    #Find the center of sample
    def drawCentral(self,circleList, centerOfImg, img):
        
        sumX = 0
        sumY = 0
        sumR = 0
        avgX = 0
        avgY = 0
        avgR = 0
        count = 0
        print "ImgCenter" + str(centerOfImg)
        (xc_tmp,yc_tmp) = centerOfImg
        #If it's a image without sample, return origin as center and radius=0
        if circleList==None:
            return(0,0),0
        #Iterate all circles can represent the boundary of sample
        #Find the center of ROI by average of the center of these circles
        #Average radius is also calculated by the way, but I did not use it.
        for circleContour in circleList:
            (x_tmp,y_tmp) = circleContour.centre
            r_tmp = circleContour.radius
            #print "Centre"  + str(circleContour.centre)
            #print "Radius" + str(circleContour.radius)
            if x_tmp>1.2*xc_tmp or x_tmp<0.8*xc_tmp:
                continue
            if y_tmp>1.2*yc_tmp or y_tmp<0.8*yc_tmp:
                continue
            sumX = sumX + x_tmp
            sumY = sumY + y_tmp
            sumR = sumR + r_tmp
            count = count + 1
        if count>0:
            print "Number of circles about our sample" + str(count)
            avgX = (sumX + count*xc_tmp)/(count*2)
            avgY = (sumY + count*yc_tmp)/(count*2)
            avgR = sumR/count
        
            #Draw ROI on copy of origin image
            p1 = (int(avgX-avgY*0.35),int(avgY*0.65))
            p2 = (int(avgX+avgY*0.35),int(avgY*1.35))
            p3 = (int(avgX-avgY*0.4),int(avgY*0.6))
            p4 = (int(avgX+avgY*0.4),int(avgY*1.4))
            cv2.rectangle(img, p1, p2, (255,255,255),2,8,0)
            cv2.rectangle(img, p3, p4, (255,255,255),2,8,0)
            #cv2.circle(img,(avgX,avgY),int(0.3*avgR),(0,0,255),2,8,0)
            #cv2.circle(img,(avgX,avgY),int(0.6*avgR),(0,0,255),2,8,0)
            
    
            return (avgX,avgY),avgR
        else:
            return(0,0),0
            
        
        

    #Calculate mean and Standard variance
    def meanStdVarInsideROI(self,ROI):
        mean, stdVar = cv2.meanStdDev(ROI)
        return mean, stdVar
        
    
    def sampleDetectResult(self,ROIoutside,ROIMid,img,avgX,avgY):
        if len(ROIMid)<2:
            print "It is not a sample"
            dataRow = [self.myid,0,0,0,0,-50,3]
            return dataRow
        else:
            
            print (len(ROIoutside))
            
            #Calculate mean and standard variance in two kinds of ROI(actually done by iterate pixels on diagonal)
            #Features: midVar and outsideVar
            #the XXX samples have both large midVar and outsideVar
            #the XXX samples have quite small |midVar-outsideVar|
            midMean, midVar = self.meanStdVarInsideROI(ROIMid)
            print midMean, midVar
            outsideMean, outsideVar = self.meanStdVarInsideROI(ROIoutside)
            print outsideMean, outsideVar
            retOut, threshOut = cv2.threshold(ROIoutside,outsideMean,255,cv2.THRESH_TOZERO)
            
            #If var is large, I assume it's a XXX sample, use larger threshold is OK.
            #Otherwise, use a threshold a litter larger than average intensity to get binary ROI
            if outsideVar>16:
                retOut1, threshOut1 = cv2.threshold(threshOut,outsideMean+30,255,cv2.THRESH_BINARY)
                
            else:
                retOut1, threshOut1 = cv2.threshold(threshOut,outsideMean+15,255,cv2.THRESH_BINARY)
               
            #Back up for test
            threshOut1BackUp = threshOut1
            
            
            #Find contour
            contoursOut1, hierachyOut_1 = cv2.findContours(threshOut1,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            #Features: Average Product
            #Average Product = sum(myDistSquare*radius)/countCircles
            #Average Product is very reliable for recognize any kind of samples from the other two kinds
            
            #Optional Features
            
            #Features: Average Ratio
            #Average Ratio = sum(myDistSquare/radius)/countCircles
            #Average ratio can differentiate good sample from the other two kinds well
            
            #Features: Large Circle Ratio
            #Large Circle Ratio = countLargeCircles/countCircles
            #Average ratio can differentiate XXX sample from the other two kinds well
            countCircles = 0
            countLargeCircles = 0
            totalProduct = 0
            totalRatio = 0
            
            #Get center of ROI
            xc_tmp = int(len(ROIoutside)/2)
            yc_tmp = int(len(ROIoutside[0])/2)
            cv2.circle(img, (avgX,avgY), 50, (0,0,255),-1)
            for indice in range(0,len(contoursOut1)):
                
                    #Use circleContour to represent white area in the binary ROI image
                    myCircleContour = circleContour([],(0,0),0)
                    
                    #Simplify contours and find minEnclosingCircle for contours
                    myCircleContour.contour = cv2.approxPolyDP(contoursOut1[indice], 0.05*cv2.arcLength(contoursOut1[indice],True), True)
                    (x_tmp,y_tmp), circleContourRadius_tmp = cv2.minEnclosingCircle(myCircleContour.contour)
                    countCircles = countCircles + 1
                    myCircleContour.centre = (int(x_tmp),int(y_tmp))
                    myCircleContour.radius = int(circleContourRadius_tmp)
                    
                    #Calculation to get features
                    if(myCircleContour.radius > xc_tmp/10):
                        countLargeCircles = countLargeCircles + 1
                    myDistSquare = ((x_tmp-xc_tmp)**2+(y_tmp-yc_tmp)**2)
                    
                    totalProduct = totalProduct + myDistSquare*myCircleContour.radius
                    totalRatio = totalRatio + myDistSquare**(0.5)/myCircleContour.radius
                    centerForDraw = (int(x_tmp-avgY*0.4+avgX),int(y_tmp+0.6*avgY))
                    
                    #Draw the circles
                    cv2.circle(img,centerForDraw , myCircleContour.radius,(255,0,0),2)
                    

            
            avgProduct = totalProduct/float(countCircles)
            avgRatio = totalRatio/float(countCircles)
            largeCirclePercent = float(countLargeCircles)/float(countCircles)*100
            #print "AVG PRODUCT = "  + str(avgProduct)
            #print "AVG Ratio = " + str(avgRatio)
            #print "LAG ratio = " + str(largeCirclePercent)
            
            #Optional features are not used for manual decision boundary
            detectResult = -1
            if avgProduct < 150000 and outsideVar < 13:
                detectResult = 0
                print "SAMPLE!!!!!!"
            elif midVar > max(18,outsideVar - 3) and outsideVar > 18:
                
                if avgProduct > 800000:
                    detectResult = 2
                    print "XXX SAMPLE"
                else:
                    detectResult = 1
                    print "X SAMPLE"
            else:
                detectResult = 1
                print "X SAMPLE"
            
            
            
            print str(self.myid)
            
            cv2.imwrite("/home/mcspinemo_guo/Downloads/Albert Wong/display/"+str(self.myid)+"afterProc.jpg",img)
            dataRow = [self.myid,str(avgProduct),str(midVar[0][0]),str(outsideVar[0][0]),avgRatio,largeCirclePercent,detectResult]
            return dataRow
           
    
    def go(self):
        
        #Read image from file
        if self.myid >=10:
            mypath = "/home/mcspinemo_guo/Downloads/Albert Wong/Syphilis Test Jpeg/DSC_68"+str(self.myid)+".JPG"
        else:
            mypath = "/home/mcspinemo_guo/Downloads/Albert Wong/Syphilis Test Jpeg/DSC_680"+str(self.myid)+".JPG"
        
        im = cv2.imread(mypath)
        if im == None:
            return
        width = len(im[0])
        length = len(im)
        
        centreOfImg = (width/2,length/2)
        #copy of the image used to show features
        icircle = cv2.imread(mypath)
        
        #use HSV channel to get the features
        imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        
        hue, saturation, value = cv2.split(imhsv)
        
        #All the features are extracted from the saturation channel
        #Turn the image from saturation to binary by threshold
        
        #First remove the Marker pen's ink from the image
        retS0, threshS0 = cv2.threshold(saturation, 120, 255, cv2.THRESH_TOZERO_INV)
        
        #Then, turn the picture into binary
        retS,threshS = cv2.threshold(threshS0,6,255,cv2.THRESH_BINARY)
        
        #Boundary enhancement by |dilation-erosion|, prepare for findContours
        erodeelement = cv2.getStructuringElement(cv2.MORPH_ERODE,(9,9))
        dilateelement = cv2.getStructuringElement(cv2.MORPH_DILATE,(5,5))
        erodeimgS = cv2.erode(threshS,erodeelement)
        dilateimgS = cv2.dilate(threshS,dilateelement)
        absimgS = cv2.absdiff(erodeimgS, dilateimgS)
        
        #findContours prefers to find white things out of black background, so I NOT it  
        absimg_bar = cv2.bitwise_not(absimgS)
        
        #Find the contour of the cells and use circle contours to represent the boundary of the sample  
        circleList = self.drawContoursWithReasonableRadiusByCircle(absimg_bar, icircle)
        
        #Find the center of the sample
        (avgX,avgY),avgR = self.drawCentral(circleList, centreOfImg, threshS0)
        
        #Get ROI, parameters are result of test.
        #Two ROI used for extract different features.
    
        imgROI_Mid = saturation[int(avgY*0.65):int(avgY*1.35),int(avgX-avgY*0.35):int(avgX+avgY*0.35)]
        imgROI_Outside = saturation[int(avgY*0.6):int(avgY*1.4),int(avgX-avgY*0.4):int(avgX+avgY*0.4)]
        
        #Use ROI to extract features and classify the samples by manual decision boundary
        return self.sampleDetectResult(imgROI_Outside,imgROI_Mid,icircle,avgX,avgY)
        
        
def main():
    
    #Generate Train Data
    #I did not include the optional features
    #You can include them by modify the code
    #4 images are used for generate test data: 6808,6816,6856,6891 You can modify it
    
    trainPath = "/home/mcspinemo_guo/Downloads/Albert Wong/trainLala.csv"
    with open(trainPath, 'wb') as f:
        writer = csv.writer(f,delimiter=',')
        writer.writerow(['id','avg product' , 'mid StdVar','long StdVar','avg Ratio','Large Circle Percentage','result'])
                    
        for myid in range(0,95):
            if myid == 8:
                continue
            if myid == 16:
                continue
            if myid == 56:
                continue
            if myid == 91:
                continue
            myDetector = detector(myid)
            rowData = myDetector.go()
            if rowData != None:
                writer.writerow(rowData)
                        
            
        f.close()
        
    
    #Generate Test Data
    testPath = "/home/mcspinemo_guo/Downloads/Albert Wong/testLala.csv"
    with open(testPath, 'wb') as f:
        writer = csv.writer(f,delimiter=',')
        writer.writerow(['id','avg Product' , 'mid StdVar','long StdVar','avg Ratio','Large Circle Percentage'])
        testList = [8,16,56,91]
        for myid in testList:
            myDetector = detector(myid)
            rowData = myDetector.go()
            if rowData != None:
                writer.writerow(rowData)
                        
            
        f.close()

if __name__=="__main__":
    main()
